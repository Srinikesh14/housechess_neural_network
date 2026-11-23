"""Stockfish position evaluation and a persistent, settings-aware SQLite cache."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine

MATE_TARGET_FLOOR = 0.9


@dataclass(frozen=True)
class PositionEvaluation:
    """A White-perspective teacher score and its network-scale target."""

    evaluation_type: str
    score: float
    target: float


def normalize_centipawns(score_cp: float, scale: float = 400.0) -> float:
    if not math.isfinite(score_cp) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Centipawn score must be finite and scale must be positive.")
    return math.tanh(score_cp / scale)


def normalize_mate(mate_plies: int) -> float:
    """Map forced mate to +/-0.9..1; shorter mates receive larger magnitude."""
    sign = 1.0 if mate_plies >= 0 else -1.0
    return sign * (MATE_TARGET_FLOOR + (1.0 - MATE_TARGET_FLOOR) / (abs(mate_plies) + 1))


def position_key(board: chess.Board) -> str:
    """Hash the FEN-relevant state and available move history for repetition context."""
    history = hashlib.sha256(
        " ".join(move.uci() for move in board.move_stack).encode("ascii")
    ).hexdigest()
    state = (
        board.board_fen(),
        int(board.turn),
        int(board.castling_rights),
        board.ep_square,
        board.halfmove_clock,
        history,
    )
    return hashlib.sha256(json.dumps(state, separators=(",", ":")).encode("ascii")).hexdigest()


class EvaluationCache:
    """SQLite cache keyed by position and the exact Stockfish search profile."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                position_key TEXT NOT NULL,
                settings_key TEXT NOT NULL,
                evaluation_type TEXT NOT NULL,
                score REAL NOT NULL,
                search_depth INTEGER,
                time_limit REAL,
                threads INTEGER NOT NULL,
                hash_mb INTEGER NOT NULL,
                engine_signature TEXT NOT NULL,
                perspective TEXT NOT NULL,
                normalized_target REAL NOT NULL,
                target_scale REAL NOT NULL,
                PRIMARY KEY (position_key, settings_key)
            )
            """
        )
        self.connection.commit()

    def get(
        self, key: str, settings_key: str, target_scale: float
    ) -> PositionEvaluation | None:
        row = self.connection.execute(
            """
            SELECT evaluation_type, score, normalized_target, target_scale
            FROM evaluations
            WHERE position_key = ? AND settings_key = ? AND perspective = 'white'
            """,
            (key, settings_key),
        ).fetchone()
        if row is None:
            return None

        evaluation_type, score, cached_target, cached_scale = row
        target = (
            float(cached_target)
            if float(cached_scale) == target_scale
            else target_for(evaluation_type, float(score), target_scale)
        )
        if float(cached_scale) != target_scale:
            self.connection.execute(
                """
                UPDATE evaluations
                SET normalized_target = ?, target_scale = ?
                WHERE position_key = ? AND settings_key = ?
                """,
                (target, target_scale, key, settings_key),
            )
            self.connection.commit()
        return PositionEvaluation(evaluation_type, float(score), target)

    def put(
        self,
        key: str,
        settings_key: str,
        record: PositionEvaluation,
        target_scale: float,
        search_depth: int | None,
        time_limit: float | None,
        threads: int,
        hash_mb: int,
        engine_signature: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO evaluations (
                position_key, settings_key, evaluation_type, score, search_depth,
                time_limit, threads, hash_mb, engine_signature, perspective,
                normalized_target, target_scale
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'white', ?, ?)
            """,
            (
                key,
                settings_key,
                record.evaluation_type,
                record.score,
                search_depth,
                time_limit,
                threads,
                hash_mb,
                engine_signature,
                record.target,
                target_scale,
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> EvaluationCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def target_for(evaluation_type: str, score: float, scale: float) -> float:
    if evaluation_type == "centipawn":
        return normalize_centipawns(score, scale)
    if evaluation_type == "mate":
        return normalize_mate(int(score))
    raise ValueError(f"Unknown evaluation type: {evaluation_type}")


class StockfishEvaluationError(RuntimeError):
    """An engine evaluation failed and the position cannot be labeled."""


class StockfishEvaluator:
    """Own one UCI engine process and reuse it across positions."""

    def __init__(
        self,
        executable: str | Path,
        cache_path: str | Path,
        *,
        depth: int | None = 8,
        time_limit: float | None = None,
        threads: int = 1,
        hash_mb: int = 64,
        cp_scale: float = 400.0,
        cache: EvaluationCache | None = None,
        engine: Any | None = None,
    ) -> None:
        if depth is None and time_limit is None:
            raise ValueError("Provide a Stockfish depth or time limit.")
        if depth is not None and depth <= 0:
            raise ValueError("Stockfish depth must be positive.")
        if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
            raise ValueError("Stockfish time limit must be positive and finite.")
        if threads <= 0 or hash_mb <= 0:
            raise ValueError("Stockfish threads and hash memory must be positive.")
        if not math.isfinite(cp_scale) or cp_scale <= 0.0:
            raise ValueError("Centipawn normalization scale must be positive and finite.")

        self.executable = str(executable)
        self.depth = depth
        self.time_limit = time_limit
        self.threads = threads
        self.hash_mb = hash_mb
        self.cp_scale = cp_scale
        self.cache = cache or EvaluationCache(cache_path)
        self._owns_cache = cache is None
        self.engine = engine
        self._owns_engine = engine is None
        self.settings_key: str | None = None
        self.engine_signature: str | None = None
        self.cache_hits = 0
        self.new_evaluations = 0
        self.failures = 0

    def start(self) -> None:
        if self.engine is None:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.executable)
            except (OSError, chess.engine.EngineError) as error:
                raise StockfishEvaluationError(
                    f"Could not start Stockfish at {self.executable!r}: {error}"
                ) from error
        try:
            self.engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
        except (OSError, chess.engine.EngineError) as error:
            raise StockfishEvaluationError(f"Could not configure Stockfish: {error}") from error

        reported_id = getattr(self.engine, "id", {})
        try:
            executable_path = str(Path(self.executable).resolve())
        except OSError:
            executable_path = self.executable
        self.engine_signature = json.dumps(
            {
                "path": executable_path,
                "id": reported_id,
            },
            sort_keys=True,
            default=str,
        )
        profile = {
            "engine": self.engine_signature,
            "depth": self.depth,
            "time_limit": self.time_limit,
            "threads": self.threads,
            "hash_mb": self.hash_mb,
        }
        self.settings_key = hashlib.sha256(
            json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def evaluate(self, board: chess.Board) -> PositionEvaluation:
        if self.engine is None or self.settings_key is None:
            self.start()

        assert self.settings_key is not None
        key = position_key(board)
        cached = self.cache.get(key, self.settings_key, self.cp_scale)
        if cached is not None:
            self.cache_hits += 1
            return cached

        assert self.engine is not None
        try:
            info = self.engine.analyse(
                board,
                chess.engine.Limit(depth=self.depth, time=self.time_limit),
            )
            score = info["score"].pov(chess.WHITE)
            mate_plies = score.mate()
            if mate_plies is not None:
                if mate_plies == 0:
                    signed_mate = score.score(mate_score=100_000)
                    mate_plies = (
                        1 if signed_mate is not None and signed_mate > 0 else -1
                    )
                evaluation_type = "mate"
                raw_score = float(mate_plies)
            else:
                centipawns = score.score()
                if centipawns is None:
                    raise StockfishEvaluationError("Stockfish returned no usable score.")
                evaluation_type = "centipawn"
                raw_score = float(centipawns)
        except (OSError, TimeoutError, chess.engine.EngineError, KeyError) as error:
            self.failures += 1
            raise StockfishEvaluationError(f"Stockfish evaluation failed: {error}") from error

        record = PositionEvaluation(
            evaluation_type,
            raw_score,
            target_for(evaluation_type, raw_score, self.cp_scale),
        )
        assert self.engine_signature is not None
        self.cache.put(
            key,
            self.settings_key,
            record,
            self.cp_scale,
            self.depth,
            self.time_limit,
            self.threads,
            self.hash_mb,
            self.engine_signature,
        )
        self.new_evaluations += 1
        return record

    def close(self) -> None:
        if self._owns_engine and self.engine is not None:
            self.engine.quit()
            self.engine = None
        if self._owns_cache:
            self.cache.close()

    def __enter__(self) -> StockfishEvaluator:
        try:
            self.start()
        except StockfishEvaluationError:
            if self._owns_cache:
                self.cache.close()
            raise
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
