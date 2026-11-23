"""Controlled LLM-assisted weight-space experiments for the V2 NumPy model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np

from backend.evaluate import GameResults, _play_model_game, _play_models
from backend.encoder import INPUT_SIZE, encode_board
from backend.mlp import MLP
from backend.stockfish_eval import StockfishEvaluationError, StockfishEvaluator
from backend.train import (
    DEFAULT_CACHE,
    DEFAULT_PGN,
    _iter_pgn,
    _sampled_boards,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(__file__).with_name("models") / "model_v2_stockfish.npz"
DEFAULT_EXPERIMENTS = Path(__file__).with_name("wsp_experiments")
DEFAULT_DATA = Path(__file__).with_name("wsp_data")
DEFAULT_MODEL_ID = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_DATASET_POSITIONS = 5000
MIN_ACCEPTANCE_POSITIONS = 100
MAX_EDIT_ABS_DELTA = 0.02
MAX_EDIT_COORDINATES = 64


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.stem}-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def prepare_datasets(
    pgn_path: Path,
    output_dir: Path,
    *,
    stockfish: str | Path,
    cache_path: Path = DEFAULT_CACHE,
    max_games: int | None = 420_000,
    development_after_game: int = 400_000,
    sealed_after_game: int = 410_000,
    max_positions_per_split: int = 2_000,
    samples_per_game: int = 4,
    seed: int = 42,
    depth: int = 8,
    time_limit: float | None = None,
    threads: int = 1,
    hash_mb: int = 64,
    cp_scale: float = 400.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create game-disjoint development and sealed datasets from held-out games."""
    pgn_path = Path(pgn_path)
    output_dir = Path(output_dir)
    if not pgn_path.is_file():
        raise FileNotFoundError(f"PGN not found: {pgn_path}")
    if max_games is not None and max_games <= 0:
        raise ValueError("--max-games must be positive.")
    if development_after_game < 0 or sealed_after_game <= development_after_game:
        raise ValueError(
            "--sealed-after-game must be greater than --development-after-game."
        )
    if max_games is not None and max_games <= sealed_after_game:
        raise ValueError(
            "--max-games must extend beyond --sealed-after-game to include test games."
        )
    if not 1 <= max_positions_per_split <= MAX_DATASET_POSITIONS:
        raise ValueError(
            f"--max-positions-per-split must be between 1 and {MAX_DATASET_POSITIONS}."
        )
    if samples_per_game <= 0:
        raise ValueError("Sample count must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        "development": output_dir / "development.jsonl",
        "sealed": output_dir / "sealed.jsonl",
    }
    if not overwrite and any(path.exists() for path in destinations.values()):
        raise FileExistsError(
            f"A WSP dataset already exists in {output_dir}; use --overwrite-data "
            "only when intentionally replacing it."
        )

    temporary_paths: dict[str, Path] = {}
    handles: dict[str, Any] = {}
    counts = {"development": 0, "sealed": 0}
    attempted = {"development": 0, "sealed": 0}
    skipped = {"development": 0, "sealed": 0}
    try:
        for split, destination in destinations.items():
            temporary = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{destination.stem}-",
                suffix=".tmp",
                dir=output_dir,
                delete=False,
            )
            temporary_paths[split] = Path(temporary.name)
            handles[split] = temporary

        with StockfishEvaluator(
            stockfish,
            cache_path,
            depth=depth,
            time_limit=time_limit,
            threads=threads,
            hash_mb=hash_mb,
            cp_scale=cp_scale,
        ) as teacher:
            for game_number, game in _iter_pgn(pgn_path):
                if max_games is not None and game_number > max_games:
                    break
                if game_number % 5 != 0 or game.errors:
                    continue
                if game_number <= development_after_game:
                    continue
                split = (
                    "development"
                    if game_number <= sealed_after_game
                    else "sealed"
                )
                if counts[split] >= max_positions_per_split:
                    if all(
                        count >= max_positions_per_split for count in counts.values()
                    ):
                        break
                    continue
                try:
                    boards = _sampled_boards(
                        game,
                        game_number,
                        samples_per_game,
                        seed + 1_000_003,
                    )
                except (ValueError, chess.IllegalMoveError):
                    continue
                for position_index, board in enumerate(boards):
                    if counts[split] >= max_positions_per_split:
                        break
                    attempted[split] += 1
                    try:
                        score = teacher.evaluate(board)
                    except StockfishEvaluationError as error:
                        skipped[split] += 1
                        if skipped[split] == 1:
                            print(f"Skipping failed Stockfish evaluation: {error}", flush=True)
                        continue
                    record = {
                        "game_id": game_number,
                        "position_id": position_index,
                        "fen": board.fen(en_passant="fen"),
                        "evaluation_type": score.evaluation_type,
                        "score": score.score,
                        "target": score.target,
                    }
                    handles[split].write(json.dumps(record, allow_nan=False) + "\n")
                    counts[split] += 1
                if game_number % 5000 == 0:
                    print(
                        f"Scanned {game_number:,} games; "
                        f"development={counts['development']:,}, "
                        f"sealed={counts['sealed']:,}",
                        flush=True,
                    )

            cache_hits = teacher.cache_hits
            new_evaluations = teacher.new_evaluations
            engine_signature = teacher.engine_signature

        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        empty_splits = [split for split, count in counts.items() if count == 0]
        if empty_splits:
            raise ValueError(
                "No positions were produced for: " + ", ".join(empty_splits)
                + "; increase --max-games or check the PGN."
            )
        for split, destination in destinations.items():
            os.replace(temporary_paths[split], destination)
    finally:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        for temporary_path in temporary_paths.values():
            if temporary_path.exists():
                temporary_path.unlink()

    manifest: dict[str, Any] = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pgn_path": str(pgn_path.resolve()),
        "pgn_size": pgn_path.stat().st_size,
        "pgn_mtime_ns": pgn_path.stat().st_mtime_ns,
        "max_games": max_games,
        "development_after_game": development_after_game,
        "sealed_after_game": sealed_after_game,
        "samples_per_game": samples_per_game,
        "seed": seed,
        "split_rule": (
            "Only parsed games divisible by 5 and with no PGN errors are included. "
            "Games after development_after_game and through sealed_after_game are "
            "development; later games are sealed. Earlier games are excluded."
        ),
        "engine_signature": engine_signature,
        "search": {
            "depth": depth,
            "time_limit": time_limit,
            "threads": threads,
            "hash_mb": hash_mb,
            "cp_scale": cp_scale,
        },
        "cache_hits": cache_hits,
        "new_stockfish_evaluations": new_evaluations,
        "counts": counts,
        "attempted": attempted,
        "skipped": skipped,
        "datasets": {},
    }
    for split, destination in destinations.items():
        manifest["datasets"][split] = {
            "path": destination.name,
            "sha256": _sha256_file(destination),
            "positions": counts[split],
            "games": len(
                {
                    json.loads(line)["game_id"]
                    for line in destination.read_text(encoding="utf-8").splitlines()
                }
            ),
        }
    _atomic_json(output_dir / "manifest.json", manifest)
    sealed_lock = output_dir / "sealed_evaluation.lock"
    if sealed_lock.exists():
        sealed_lock.unlink()
    print(
        f"Created WSP datasets: development={counts['development']:,}, "
        f"sealed={counts['sealed']:,}; cache hits={cache_hits:,}, "
        f"new evaluations={new_evaluations:,}, "
        f"skipped={sum(skipped.values()):,}",
        flush=True,
    )
    return manifest


def load_dataset(path: Path, limit: int = MAX_DATASET_POSITIONS) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if limit <= 0 or limit > MAX_DATASET_POSITIONS:
        raise ValueError(f"Dataset limit must be between 1 and {MAX_DATASET_POSITIONS}.")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            if len(records) >= limit:
                raise ValueError(
                    f"Dataset exceeds the {limit}-position tool limit; prepare a smaller "
                    "dataset or increase the reviewed limit in the code."
                )
            try:
                record = json.loads(line)
                board = chess.Board(record["fen"])
                target = float(record["target"])
                game_id = int(record["game_id"])
                position_id = int(record["position_id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid dataset record on line {line_number}: {error}") from error
            if not board.is_valid():
                raise ValueError(f"Invalid chess position on dataset line {line_number}.")
            if not math.isfinite(target) or not -1.0 <= target <= 1.0:
                raise ValueError(f"Invalid target on dataset line {line_number}.")
            records.append(
                {
                    "fen": record["fen"],
                    "target": target,
                    "game_id": game_id,
                    "position_id": position_id,
                }
            )
    if not records:
        raise ValueError(f"Dataset contains no positions: {path}")
    return records


def _require_dataset_split(path: Path, expected_split: str) -> dict[str, Any]:
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"Missing dataset manifest next to {path}; prepare datasets with "
            "`python -m backend.wsp prepare-data`."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_info = manifest.get("datasets", {}).get(expected_split)
    if (
        not isinstance(dataset_info, dict)
        or dataset_info.get("path") != path.name
        or dataset_info.get("sha256") != _sha256_file(path)
    ):
        raise ValueError(
            f"{path} is not the unchanged {expected_split} dataset recorded in its manifest."
        )
    return manifest


def _prediction_metrics(
    model: MLP, records: list[dict[str, Any]]
) -> dict[str, Any]:
    predictions: list[float] = []
    targets: list[float] = []
    started = time.perf_counter()
    for record in records:
        board = chess.Board(record["fen"])
        predictions.append(model.forward(encode_board(board)))
        targets.append(float(record["target"]))
    elapsed = time.perf_counter() - started
    predicted = np.asarray(predictions, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    errors = predicted - expected
    correlation: float | None = None
    if predicted.size > 1 and np.std(predicted) > 0.0 and np.std(expected) > 0.0:
        correlation = float(np.corrcoef(predicted, expected)[0, 1])
    return {
        "positions": int(predicted.size),
        "mse": float(np.mean(errors**2)),
        "mae": float(np.mean(np.abs(errors))),
        "correlation": correlation,
        "max_absolute_error": float(np.max(np.abs(errors))),
        "runtime_ms_per_position": elapsed * 1000.0 / predicted.size,
    }


def _validate_model(model: MLP) -> None:
    if (
        model.W1.shape != (INPUT_SIZE, 256)
        or model.b1.shape != (256,)
        or model.W2.shape != (256, 64)
        or model.b2.shape != (64,)
        or model.W3.shape != (64, 1)
        or model.b3.shape != (1,)
    ):
        raise ValueError("WSP only supports the trained 838 -> 256 -> 64 -> 1 V2 model.")


def _array_summary(values: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).ravel()
    return {
        "shape": list(values.shape),
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "quantiles_01_50_99": [
            float(value) for value in np.quantile(flat, [0.01, 0.5, 0.99])
        ],
        "non_finite_count": int(np.count_nonzero(~np.isfinite(flat))),
    }


class WSPTools:
    """Bounded local analysis tools; no filesystem or code-execution tool exists."""

    def __init__(
        self,
        model: MLP,
        records: list[dict[str, Any]],
        *,
        proposal_path: Path | None = None,
        allow_proposal: bool = False,
    ) -> None:
        _validate_model(model)
        self.model = model
        self.records = records
        self.proposal_path = proposal_path
        self.allow_proposal = allow_proposal
        self.proposal: dict[str, Any] | None = None

    def available_tools(self) -> list[dict[str, Any]]:
        tools = [
            {
                "name": "inspect_architecture",
                "arguments": {},
                "purpose": "Return layer dimensions, activations, parameter count and model version.",
            },
            {
                "name": "layer_statistics",
                "arguments": {"parameter": "W1 | b1 | W2 | b2 | W3 | b3"},
                "purpose": "Return bounded summary statistics for one parameter array.",
            },
            {
                "name": "inspect_neuron",
                "arguments": {
                    "layer": "hidden1 | hidden2",
                    "neuron_index": "integer",
                },
                "purpose": "Inspect top incoming and outgoing weights for one hidden neuron.",
            },
            {
                "name": "analyze_activations",
                "arguments": {
                    "layer": "hidden1 | hidden2",
                    "neuron_index": "integer",
                    "max_positions": "integer from 1 to 512",
                },
                "purpose": "Measure a neuron activation across the fixed local dataset.",
            },
            {
                "name": "find_similar_neurons",
                "arguments": {
                    "layer": "hidden1 | hidden2",
                    "neuron_index": "integer",
                },
                "purpose": "Find neurons with similar incoming weight vectors; similarity is not semantic proof.",
            },
            {
                "name": "evaluate_positions",
                "arguments": {"position_ids": "list of at most 32 zero-based dataset indices"},
                "purpose": "Compare baseline prediction with stored Stockfish targets for selected positions.",
            },
            {
                "name": "analyze_errors",
                "arguments": {},
                "purpose": "Summarize absolute errors and return at most 10 largest-error examples.",
            },
        ]
        if self.allow_proposal:
            tools.append(
                {
                    "name": "propose_weight_edit",
                    "arguments": {
                        "parameter": "W1 | b1 | W2 | b2 | W3 | b3",
                        "coordinates": "list of at most 64 index lists",
                        "operation": "add | set",
                        "value": "finite numeric value",
                        "reason": "evidence-based explanation",
                        "expected_effect": "testable expected effect",
                        "potential_side_effects": "risks",
                    },
                    "purpose": "Validate and record a tiny edit proposal only; does not change any model.",
                }
            )
        return tools

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        dispatch = {
            "inspect_architecture": self.inspect_architecture,
            "layer_statistics": self.layer_statistics,
            "inspect_neuron": self.inspect_neuron,
            "analyze_activations": self.analyze_activations,
            "find_similar_neurons": self.find_similar_neurons,
            "evaluate_positions": self.evaluate_positions,
            "analyze_errors": self.analyze_errors,
        }
        if name == "propose_weight_edit" and self.allow_proposal:
            return self.propose_weight_edit(**arguments)
        function = dispatch.get(name)
        if function is None:
            raise ValueError(f"Tool is not available: {name}")
        return function(**arguments)

    def inspect_architecture(self) -> dict[str, Any]:
        parameters = (self.model.W1, self.model.b1, self.model.W2, self.model.b2,
                      self.model.W3, self.model.b3)
        return {
            "architecture": [
                {"width": INPUT_SIZE, "meaning": "encoded chess position"},
                {"width": int(self.model.W1.shape[1]), "activation": "ReLU"},
                {"width": int(self.model.W2.shape[1]), "activation": "ReLU"},
                {"width": 1, "activation": "tanh"},
            ],
            "parameter_shapes": {
                "W1": list(self.model.W1.shape),
                "b1": list(self.model.b1.shape),
                "W2": list(self.model.W2.shape),
                "b2": list(self.model.b2.shape),
                "W3": list(self.model.W3.shape),
                "b3": list(self.model.b3.shape),
            },
            "parameter_count": int(sum(array.size for array in parameters)),
            "input_feature_map": {
                "0-767": "12 piece planes x 64 squares; white pawn through king, then black pawn through king",
                "768": "side to move (1 for White)",
                "769-772": "white king-side, white queen-side, black king-side, black queen-side castling rights",
                "773-836": "one-hot en-passant target square",
                "837": "halfmove clock clipped to 100 and divided by 100",
            },
            "dataset_positions": len(self.records),
            "weights_are_read_only": True,
            "training_or_backprop_available": False,
        }

    def layer_statistics(self, parameter: str) -> dict[str, Any]:
        if parameter not in ("W1", "b1", "W2", "b2", "W3", "b3"):
            raise ValueError("Unknown parameter; choose W1, b1, W2, b2, W3, or b3.")
        return {"parameter": parameter, **_array_summary(getattr(self.model, parameter))}

    def inspect_neuron(self, layer: str, neuron_index: int) -> dict[str, Any]:
        if layer == "hidden1":
            width = self.model.W1.shape[1]
            if neuron_index < 0 or neuron_index >= width:
                raise ValueError(f"Neuron index out of range for {layer}.")
            incoming = self.model.W1[:, neuron_index]
            outgoing = self.model.W2[neuron_index, :]
            bias = float(self.model.b1[neuron_index])
        elif layer == "hidden2":
            width = self.model.W2.shape[1]
            if neuron_index < 0 or neuron_index >= width:
                raise ValueError(f"Neuron index out of range for {layer}.")
            incoming = self.model.W2[:, neuron_index]
            outgoing = self.model.W3[neuron_index, :]
            bias = float(self.model.b2[neuron_index])
        else:
            raise ValueError("Layer must be hidden1 or hidden2.")
        return {
            "layer": layer,
            "neuron_index": neuron_index,
            "bias": bias,
            "incoming_summary": _array_summary(incoming),
            "incoming_largest_absolute": _top_values(incoming, 12),
            "outgoing_summary": _array_summary(outgoing),
            "outgoing_largest_absolute": _top_values(outgoing, 12),
            "warning": "Weight patterns alone do not establish a neuron's chess concept.",
        }

    def analyze_activations(
        self,
        layer: str,
        neuron_index: int,
        max_positions: int = 256,
    ) -> dict[str, Any]:
        if max_positions <= 0 or max_positions > 512:
            raise ValueError("max_positions must be between 1 and 512.")
        if layer == "hidden1":
            width = self.model.W1.shape[1]
        elif layer == "hidden2":
            width = self.model.W2.shape[1]
        else:
            raise ValueError("Layer must be hidden1 or hidden2.")
        if neuron_index < 0 or neuron_index >= width:
            raise ValueError(f"Neuron index out of range for {layer}.")

        activations: list[float] = []
        targets: list[float] = []
        for record in self.records[:max_positions]:
            features = encode_board(chess.Board(record["fen"]))
            hidden1 = np.maximum(0.0, features @ self.model.W1 + self.model.b1)
            if layer == "hidden1":
                activation = float(hidden1[neuron_index])
            else:
                hidden2 = np.maximum(0.0, hidden1 @ self.model.W2 + self.model.b2)
                activation = float(hidden2[neuron_index])
            activations.append(activation)
            targets.append(float(record["target"]))
        values = np.asarray(activations, dtype=np.float64)
        labels = np.asarray(targets, dtype=np.float64)
        correlation = None
        if values.size > 1 and np.std(values) > 0.0 and np.std(labels) > 0.0:
            correlation = float(np.corrcoef(values, labels)[0, 1])
        return {
            "layer": layer,
            "neuron_index": neuron_index,
            "positions": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "active_fraction": float(np.mean(values > 0.0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "correlation_with_white_perspective_target": correlation,
            "interpretation_warning": "Correlation is descriptive, not causal or semantic proof.",
        }

    def find_similar_neurons(self, layer: str, neuron_index: int) -> dict[str, Any]:
        if layer == "hidden1":
            weight_matrix = self.model.W1
        elif layer == "hidden2":
            weight_matrix = self.model.W2
        else:
            raise ValueError("Layer must be hidden1 or hidden2.")
        if neuron_index < 0 or neuron_index >= weight_matrix.shape[1]:
            raise ValueError(f"Neuron index out of range for {layer}.")
        vectors = weight_matrix.T
        norms = np.linalg.norm(vectors, axis=1)
        denominator = norms * norms[neuron_index]
        similarities = np.zeros(vectors.shape[0], dtype=np.float64)
        valid = denominator > 0.0
        similarities[valid] = (
            vectors[valid] @ vectors[neuron_index]
        ) / denominator[valid]
        similarities[neuron_index] = -np.inf
        closest = np.argsort(similarities)[-5:][::-1]
        return {
            "layer": layer,
            "neuron_index": neuron_index,
            "similarity_measure": "cosine similarity of incoming weight vectors",
            "nearest": [
                {"neuron_index": int(index), "cosine_similarity": float(similarities[index])}
                for index in closest
            ],
            "warning": "Weight-vector similarity does not imply matching activation behavior.",
        }

    def evaluate_positions(self, position_ids: list[int]) -> dict[str, Any]:
        if not isinstance(position_ids, list) or not 1 <= len(position_ids) <= 32:
            raise ValueError("Provide 1 to 32 position_ids.")
        results = []
        for position_id in position_ids:
            if isinstance(position_id, bool) or not isinstance(position_id, int):
                raise ValueError("Position IDs must be integers.")
            if position_id < 0 or position_id >= len(self.records):
                raise ValueError(f"Position ID out of range: {position_id}.")
            record = self.records[position_id]
            prediction = self.model.forward(
                encode_board(chess.Board(record["fen"]))
            )
            results.append(
                {
                    "position_id": position_id,
                    "fen": record["fen"],
                    "prediction": prediction,
                    "stockfish_target": record["target"],
                    "signed_error": prediction - record["target"],
                }
            )
        return {"positions": results}

    def analyze_errors(self) -> dict[str, Any]:
        scored: list[tuple[float, int, float, float]] = []
        buckets = {
            "near_equal": [],
            "moderate_advantage": [],
            "large_advantage": [],
        }
        for index, record in enumerate(self.records):
            prediction = self.model.forward(
                encode_board(chess.Board(record["fen"]))
            )
            target = float(record["target"])
            error = prediction - target
            scored.append((abs(error), index, prediction, target))
            magnitude = abs(target)
            category = (
                "near_equal" if magnitude < 0.2
                else "moderate_advantage" if magnitude < 0.7
                else "large_advantage"
            )
            buckets[category].append(error * error)
        return {
            "count": len(scored),
            "mse_by_target_magnitude": {
                key: float(np.mean(values)) if values else None
                for key, values in buckets.items()
            },
            "largest_errors": [
                {
                    "position_id": index,
                    "absolute_error": magnitude,
                    "prediction": prediction,
                    "target": target,
                }
                for magnitude, index, prediction, target
                in sorted(scored, reverse=True)[:10]
            ],
        }

    def propose_weight_edit(
        self,
        parameter: str,
        coordinates: list[list[int]],
        operation: str,
        value: float,
        reason: str,
        expected_effect: str,
        potential_side_effects: str,
    ) -> dict[str, Any]:
        if not self.allow_proposal:
            raise ValueError("Weight-edit proposals are disabled in this run.")
        if self.proposal is not None:
            raise ValueError("Only one edit proposal is allowed per experiment.")
        if parameter not in ("W1", "b1", "W2", "b2", "W3", "b3"):
            raise ValueError("Unknown parameter array.")
        if operation not in ("add", "set"):
            raise ValueError("Operation must be add or set.")
        if not isinstance(coordinates, list) or not 1 <= len(coordinates) <= MAX_EDIT_COORDINATES:
            raise ValueError(f"Provide 1 to {MAX_EDIT_COORDINATES} coordinates.")
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("Edit value must be finite.")
        for text, label in (
            (reason, "reason"),
            (expected_effect, "expected_effect"),
            (potential_side_effects, "potential_side_effects"),
        ):
            if not isinstance(text, str) or not text.strip() or len(text) > 2000:
                raise ValueError(f"{label} must be a non-empty string of at most 2000 characters.")

        array = getattr(self.model, parameter)
        normalized_coordinates: list[list[int]] = []
        changes: list[float] = []
        for coordinate in coordinates:
            if not isinstance(coordinate, list) or len(coordinate) != array.ndim:
                raise ValueError(f"Each coordinate must have {array.ndim} indices.")
            if any(isinstance(index, bool) or not isinstance(index, int) for index in coordinate):
                raise ValueError("Coordinate indices must be integers.")
            if any(index < 0 or index >= array.shape[axis] for axis, index in enumerate(coordinate)):
                raise ValueError(f"Coordinate is out of bounds for {parameter}.")
            normalized = [int(index) for index in coordinate]
            if normalized in normalized_coordinates:
                raise ValueError("Duplicate coordinates are not allowed.")
            normalized_coordinates.append(normalized)
            old_value = float(array[tuple(normalized)])
            new_value = old_value + float(value) if operation == "add" else float(value)
            change = new_value - old_value
            if not math.isfinite(new_value) or abs(change) > MAX_EDIT_ABS_DELTA:
                raise ValueError(
                    f"Every individual change must be finite and at most "
                    f"{MAX_EDIT_ABS_DELTA} in absolute value."
                )
            changes.append(change)

        self.proposal = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "baseline_sha256": None,
            "parameter": parameter,
            "coordinates": normalized_coordinates,
            "operation": operation,
            "value": float(value),
            "maximum_absolute_delta": max(abs(change) for change in changes),
            "reason": reason.strip(),
            "expected_effect": expected_effect.strip(),
            "potential_side_effects": potential_side_effects.strip(),
            "applied": False,
        }
        return {
            "proposal_recorded": True,
            "applied": False,
            "parameter": parameter,
            "coordinates": normalized_coordinates,
            "maximum_absolute_delta": self.proposal["maximum_absolute_delta"],
            "note": "The baseline and candidate arrays are unchanged.",
        }


def _top_values(values: np.ndarray, limit: int) -> list[dict[str, Any]]:
    flat = np.asarray(values, dtype=np.float64).ravel()
    indices = np.argsort(np.abs(flat))[-limit:][::-1]
    return [
        {"index": int(index), "value": float(flat[index])}
        for index in indices
    ]


def _project_env_value(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    requested_name = name
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            env_name, value = stripped.split("=", 1)
            if env_name.strip() == requested_name:
                value = value.strip().strip("\"'")
                if value:
                    return value
    return ""


def _load_api_key() -> str:
    key = _project_env_value("OPENROUTER_API_KEY")
    if key:
        return key
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing. Set it in the process environment or "
        "in the ignored project-root .env file."
    )


class OpenRouterClient:
    """Small OpenAI-compatible client; API keys are never written to logs."""

    def __init__(self, model: str, timeout: float = 120.0) -> None:
        if not model.strip():
            raise ValueError("Model identifier cannot be empty.")
        self.model = model
        self.timeout = timeout
        self.api_key = _load_api_key()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def complete(self, messages: list[dict[str, str]]) -> str:
        request_body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1800,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=request_body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Chess ANN Lab WSP",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            error.read()
            raise RuntimeError(
                f"OpenRouter returned HTTP {error.code}. "
                "Check the selected model, account quota, and API key."
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"OpenRouter request failed: {error}") from error
        except json.JSONDecodeError as error:
            raise RuntimeError("OpenRouter returned invalid JSON.") from error

        if not isinstance(payload, dict):
            raise RuntimeError("OpenRouter response must be a JSON object.")
        usage = payload.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        try:
            self.prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.completion_tokens += int(usage.get("completion_tokens", 0))
        except (TypeError, ValueError) as error:
            raise RuntimeError("OpenRouter returned invalid token-usage values.") from error
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                f"OpenRouter response did not contain a message: {payload!r}"
            ) from error
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenRouter returned an empty assistant message.")
        return content.strip()


def run_analysis(
    model_path: Path,
    dataset_path: Path,
    output_root: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    max_steps: int = 10,
    allow_edit_proposal: bool = False,
    question: str = "",
) -> Path:
    if not 1 <= max_steps <= 20:
        raise ValueError("--max-steps must be between 1 and 20.")
    if len(question) > 2000:
        raise ValueError("--question must be at most 2000 characters.")
    baseline_path = Path(model_path)
    baseline = MLP.load(baseline_path)
    _validate_model(baseline)
    _require_dataset_split(Path(dataset_path), "development")
    records = load_dataset(dataset_path)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    baseline_hash = _sha256_file(baseline_path)
    dataset_hash = _sha256_file(dataset_path)
    proposal_path = run_dir / "proposal.json" if allow_edit_proposal else None
    tools = WSPTools(
        baseline,
        records,
        proposal_path=proposal_path,
        allow_proposal=allow_edit_proposal,
    )
    system_prompt_path = Path(__file__).with_name("wsp_system_prompt.txt")
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    system_prompt += (
        "\n\nAvailable local tools (arguments are JSON):\n"
        + json.dumps(tools.available_tools(), indent=2)
    )
    user_prompt = json.dumps(
        {
            "task": (
                "Perform a read-only evidence-gathering investigation"
                if not allow_edit_proposal
                else "Investigate the fixed development set and, only if evidence supports it, "
                     "record one small candidate edit proposal"
            ),
            "research_question": question.strip() or (
                "Identify one evidence-supported systematic weakness, if one exists, "
                "without assuming a neuron has a chess-semantic role."
            ),
            "architecture": tools.inspect_architecture(),
            "development_set_sha256": dataset_hash,
            "baseline_model_sha256": baseline_hash,
            "first_step": "Inspect architecture, summarize baseline errors, then investigate a specific hypothesis.",
            "constraints": [
                "Use only the supplied local tools.",
                "Do not infer semantics from weight values alone.",
                "Do not request individual positions one at a time when aggregate tools suffice.",
                "The sealed test set is not available in this experiment.",
                "An edit proposal is not applied or accepted by this run.",
            ],
        },
        indent=2,
    )
    client = OpenRouterClient(model_id)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    log: list[dict[str, Any]] = []
    final_report: dict[str, Any] | None = None
    for step in range(1, max_steps + 1):
        try:
            response_text = client.complete(messages)
        except RuntimeError:
            _atomic_json(
                run_dir / "partial_log.json",
                {
                    "model": model_id,
                    "baseline_sha256": baseline_hash,
                    "dataset_sha256": dataset_hash,
                    "steps": log,
                    "token_usage": {
                        "prompt": client.prompt_tokens,
                        "completion": client.completion_tokens,
                    },
                },
            )
            raise
        log.append({"step": step, "assistant_response": response_text})
        try:
            action = json.loads(response_text)
        except json.JSONDecodeError as error:
            _atomic_json(
                run_dir / "partial_log.json",
                {
                    "model": model_id,
                    "baseline_sha256": baseline_hash,
                    "dataset_sha256": dataset_hash,
                    "steps": log,
                    "token_usage": {
                        "prompt": client.prompt_tokens,
                        "completion": client.completion_tokens,
                    },
                },
            )
            raise RuntimeError(
                "The model did not return the required JSON action. Partial log saved."
            ) from error
        if not isinstance(action, dict):
            raise RuntimeError("The model response must be a JSON object.")
        if action.get("action") == "final":
            report = action.get("report")
            if not isinstance(report, dict):
                raise RuntimeError("Final action must contain a report JSON object.")
            final_report = report
            break
        if action.get("action") != "tool" or not isinstance(action.get("name"), str):
            raise RuntimeError("Expected action=tool or action=final.")
        arguments = action.get("arguments", {})
        try:
            result = tools.dispatch(action["name"], arguments)
        except (ValueError, KeyError, TypeError) as error:
            result = {"tool_error": str(error)}
        if tools.proposal is not None and proposal_path is not None:
            proposal = dict(tools.proposal)
            proposal["baseline_sha256"] = baseline_hash
            _atomic_json(proposal_path, proposal)
        log[-1]["tool_result"] = result
        messages.extend(
            [
                {"role": "assistant", "content": response_text},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"tool_result": result}, allow_nan=False
                    ),
                },
            ]
        )

    payload = {
        "run_id": run_id,
        "model": model_id,
        "mode": "propose" if allow_edit_proposal else "read-only",
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": baseline_hash,
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": dataset_hash,
        "dataset_positions": len(records),
        "steps": log,
        "final_report": final_report,
        "termination_reason": (
            "final_report_received" if final_report is not None else "maximum_steps_reached"
        ),
        "proposal_path": str(proposal_path.resolve()) if proposal_path and proposal_path.exists() else None,
        "token_usage": {
            "prompt": client.prompt_tokens,
            "completion": client.completion_tokens,
        },
        "warning": "This is exploratory analysis, not evidence of improved playing strength.",
    }
    _atomic_json(run_dir / "experiment.json", payload)
    print(f"WSP experiment saved to {run_dir}")
    print(
        f"OpenRouter token usage (reported by provider): "
        f"prompt={client.prompt_tokens}, completion={client.completion_tokens}"
    )
    if proposal_path and proposal_path.is_file():
        print(f"Candidate edit proposal (not applied): {proposal_path}")
    return run_dir


def apply_proposal(
    proposal_path: Path,
    baseline_path: Path,
    dataset_path: Path,
    candidate_path: Path,
    *,
    approved: bool,
) -> dict[str, Any]:
    if not approved:
        raise ValueError(
            "No candidate changed. Re-run with --approve-candidate-edit after reviewing the proposal."
        )
    proposal = json.loads(Path(proposal_path).read_text(encoding="utf-8"))
    baseline_path = Path(baseline_path)
    candidate_path = Path(candidate_path)
    if candidate_path.resolve() == baseline_path.resolve():
        raise ValueError("Candidate path must not be the baseline model path.")
    if candidate_path.exists():
        raise FileExistsError(f"Candidate already exists: {candidate_path}")
    baseline_hash = _sha256_file(baseline_path)
    if proposal.get("baseline_sha256") != baseline_hash:
        raise ValueError("Proposal baseline hash does not match the selected model.")

    baseline = MLP.load(baseline_path)
    _validate_model(baseline)
    records = load_dataset(dataset_path)
    tools = WSPTools(
        baseline,
        records,
        proposal_path=None,
        allow_proposal=True,
    )
    edit = {
        key: proposal[key]
        for key in (
            "parameter",
            "coordinates",
            "operation",
            "value",
            "reason",
            "expected_effect",
            "potential_side_effects",
        )
    }
    tools.propose_weight_edit(**edit)
    validated = tools.proposal
    if validated is None:
        raise RuntimeError("Proposal validation failed without an error.")
    candidate = MLP.load(baseline_path)
    array = getattr(candidate, validated["parameter"])
    for coordinate in validated["coordinates"]:
        index = tuple(coordinate)
        if validated["operation"] == "add":
            array[index] += validated["value"]
        else:
            array[index] = validated["value"]
    if not all(
        np.all(np.isfinite(getattr(candidate, name)))
        for name in ("W1", "b1", "W2", "b2", "W3", "b3")
    ):
        raise ValueError("Candidate parameters became non-finite.")
    candidate.save(candidate_path)
    result = {
        "baseline_path": str(baseline_path.resolve()),
        "baseline_sha256": baseline_hash,
        "candidate_path": str(candidate_path.resolve()),
        "candidate_sha256": _sha256_file(candidate_path),
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": _sha256_file(Path(dataset_path)),
        "split": "development",
        "proposal": proposal,
        "baseline_metrics": _prediction_metrics(baseline, records),
        "candidate_metrics": _prediction_metrics(candidate, records),
        "accepted": False,
        "note": "Candidate was evaluated on development data only; it is not accepted.",
    }
    _atomic_json(candidate_path.with_suffix(".report.json"), result)
    return result


def evaluate_candidate(
    baseline_path: Path,
    candidate_path: Path,
    dataset_path: Path,
    output_path: Path,
    split: str,
) -> dict[str, Any]:
    if split not in ("development", "sealed"):
        raise ValueError("Split must be development or sealed.")
    _require_dataset_split(Path(dataset_path), split)
    sealed_lock = Path(dataset_path).parent / "sealed_evaluation.lock"
    if split == "sealed" and sealed_lock.exists():
        raise ValueError(
            "The sealed set has already been used for a candidate. Continue iteration "
            "on development data; reserve the sealed set for one final candidate."
        )
    baseline = MLP.load(baseline_path)
    candidate = MLP.load(candidate_path)
    _validate_model(baseline)
    _validate_model(candidate)
    records = load_dataset(dataset_path)
    report = {
        "baseline_path": str(Path(baseline_path).resolve()),
        "baseline_sha256": _sha256_file(Path(baseline_path)),
        "candidate_path": str(Path(candidate_path).resolve()),
        "candidate_sha256": _sha256_file(Path(candidate_path)),
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": _sha256_file(Path(dataset_path)),
        "split": split,
        "baseline_metrics": _prediction_metrics(baseline, records),
        "candidate_metrics": _prediction_metrics(candidate, records),
        "accepted": False,
        "note": "Evaluation does not automatically accept or deploy the candidate.",
    }
    _atomic_json(Path(output_path), report)
    if split == "sealed":
        _atomic_json(
            sealed_lock,
            {
                "baseline_sha256": report["baseline_sha256"],
                "candidate_sha256": report["candidate_sha256"],
                "dataset_sha256": report["dataset_sha256"],
                "report_path": str(Path(output_path).resolve()),
            },
        )
    return report


def accept_candidate(
    report_path: Path,
    accepted_path: Path,
    *,
    confirmed: bool,
) -> dict[str, Any]:
    if not confirmed:
        raise ValueError("Acceptance requires the explicit --confirm-accept flag.")
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("split") != "sealed":
        raise ValueError("Only a sealed-set evaluation report can qualify for acceptance.")
    baseline_path = Path(report["baseline_path"])
    candidate_path = Path(report["candidate_path"])
    if _sha256_file(baseline_path) != report.get("baseline_sha256"):
        raise ValueError("Baseline model changed since sealed evaluation.")
    if _sha256_file(candidate_path) != report.get("candidate_sha256"):
        raise ValueError("Candidate model changed since sealed evaluation.")
    dataset_path = Path(report["dataset_path"])
    _require_dataset_split(dataset_path, "sealed")
    if _sha256_file(dataset_path) != report.get("dataset_sha256"):
        raise ValueError("Sealed dataset changed since evaluation.")
    lock_path = dataset_path.parent / "sealed_evaluation.lock"
    if not lock_path.is_file():
        raise ValueError("Sealed report was not created by the final one-shot evaluation.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        lock.get("baseline_sha256") != report.get("baseline_sha256")
        or lock.get("candidate_sha256") != report.get("candidate_sha256")
        or lock.get("dataset_sha256") != report.get("dataset_sha256")
    ):
        raise ValueError("Sealed report does not match the one-shot evaluation lock.")
    baseline_metrics = report["baseline_metrics"]
    candidate_metrics = report["candidate_metrics"]
    if int(candidate_metrics.get("positions", 0)) < MIN_ACCEPTANCE_POSITIONS:
        raise ValueError(
            f"Sealed acceptance requires at least {MIN_ACCEPTANCE_POSITIONS} positions."
        )
    for metric_set in (baseline_metrics, candidate_metrics):
        for key in ("mse", "mae"):
            if not math.isfinite(float(metric_set[key])) or float(metric_set[key]) < 0.0:
                raise ValueError(f"Sealed evaluation has an invalid {key} value.")
    baseline_corr = baseline_metrics["correlation"]
    candidate_corr = candidate_metrics["correlation"]
    if baseline_corr is not None and not math.isfinite(float(baseline_corr)):
        raise ValueError("Baseline correlation is not finite.")
    if candidate_corr is not None and not math.isfinite(float(candidate_corr)):
        raise ValueError("Candidate correlation is not finite.")
    correlation_ok = (
        baseline_corr is None
        or candidate_corr is not None
        and candidate_corr >= baseline_corr - 0.01
    )
    if not (
        candidate_metrics["mse"] < baseline_metrics["mse"]
        and candidate_metrics["mae"] <= baseline_metrics["mae"] * 1.02
        and correlation_ok
    ):
        raise ValueError(
            "Candidate failed the sealed acceptance gates: MSE must improve, "
            "MAE may not regress by more than 2%, and correlation may not fall "
            "by more than 0.01."
        )
    accepted_path = Path(accepted_path)
    if accepted_path.exists():
        raise FileExistsError(
            f"Accepted model already exists: {accepted_path}; choose a new versioned path."
        )
    candidate = MLP.load(candidate_path)
    candidate.save(accepted_path)
    result = {
        "accepted_model": str(accepted_path.resolve()),
        "candidate_sha256": _sha256_file(accepted_path),
        "sealed_report": str(Path(report_path).resolve()),
        "sealed_dataset_sha256": report["dataset_sha256"],
        "metrics": candidate_metrics,
        "accepted_at": datetime.now(timezone.utc).isoformat(),
        "approval": "explicit command-line --confirm-accept",
    }
    _atomic_json(accepted_path.with_suffix(".acceptance.json"), result)
    return result


def benchmark_playing_strength(
    baseline_path: Path,
    candidate_path: Path,
    *,
    self_play_pairs: int = 2,
    stockfish_games: int = 0,
    stockfish_path: str = "stockfish",
    depth: int = 8,
    threads: int = 1,
    hash_mb: int = 64,
    max_plies: int = 300,
) -> dict[str, Any]:
    """Run paired color-swapped model games and optional Stockfish matches."""
    if self_play_pairs < 0 or stockfish_games < 0 or max_plies <= 0:
        raise ValueError("Game counts must be non-negative and max_plies positive.")
    if stockfish_games and (depth <= 0 or threads <= 0 or hash_mb <= 0):
        raise ValueError("Stockfish depth, threads, and hash must be positive.")
    baseline = MLP.load(baseline_path)
    candidate = MLP.load(candidate_path)
    _validate_model(baseline)
    _validate_model(candidate)
    report: dict[str, Any] = {
        "baseline_sha256": _sha256_file(Path(baseline_path)),
        "candidate_sha256": _sha256_file(Path(candidate_path)),
        "self_play_pairs": self_play_pairs,
        "stockfish_games_per_model": stockfish_games,
        "results": {},
        "warning": "Playing results are separate from Stockfish regression metrics.",
    }

    baseline_results = GameResults()
    candidate_results = GameResults()
    for _ in range(self_play_pairs):
        board, plies = _play_models(baseline, candidate, chess.WHITE, max_plies)
        baseline_results.add(board, chess.WHITE, plies)
        candidate_results.add(board, chess.BLACK, plies)
        board, plies = _play_models(candidate, baseline, chess.WHITE, max_plies)
        candidate_results.add(board, chess.WHITE, plies)
        baseline_results.add(board, chess.BLACK, plies)
    if self_play_pairs:
        report["results"]["baseline_vs_candidate"] = baseline_results.report()
        report["results"]["candidate_vs_baseline"] = candidate_results.report()

    if stockfish_games:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        try:
            engine.configure({"Threads": threads, "Hash": hash_mb})
            for name, model in (("baseline", baseline), ("candidate", candidate)):
                results = GameResults()
                for game_index in range(stockfish_games):
                    model_color = chess.WHITE if game_index % 2 == 0 else chess.BLACK
                    board, plies = _play_model_game(
                        model,
                        model_color,
                        engine,
                        engine_depth=depth,
                        max_plies=max_plies,
                    )
                    results.add(board, model_color, plies)
                report["results"][f"{name}_vs_stockfish"] = results.report()
        finally:
            engine.quit()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare-data", help="Build game-disjoint development and sealed Stockfish sets."
    )
    prepare.add_argument("--pgn", type=Path, default=DEFAULT_PGN)
    prepare.add_argument("--stockfish", default=os.environ.get("STOCKFISH_EXECUTABLE", "stockfish"))
    prepare.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_DATA)
    prepare.add_argument("--max-games", type=int, default=420_000)
    prepare.add_argument("--development-after-game", type=int, default=400_000)
    prepare.add_argument("--sealed-after-game", type=int, default=410_000)
    prepare.add_argument("--max-positions-per-split", type=int, default=2_000)
    prepare.add_argument("--samples-per-game", type=int, default=4)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--depth", type=int, default=8)
    prepare.add_argument("--time-limit", type=float)
    prepare.add_argument("--threads", type=int, default=1)
    prepare.add_argument("--hash-mb", type=int, default=64)
    prepare.add_argument("--cp-scale", type=float, default=400.0)
    prepare.add_argument("--overwrite-data", action="store_true")

    analyze = commands.add_parser(
        "analyze", help="Run bounded local-tool WSP analysis through OpenRouter."
    )
    analyze.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    analyze.add_argument("--dataset", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, default=DEFAULT_EXPERIMENTS)
    analyze.add_argument("--model-id", default=_project_env_value("OPENROUTER_MODEL") or DEFAULT_MODEL_ID)
    analyze.add_argument("--max-steps", type=int, default=10)
    analyze.add_argument("--propose-edit", action="store_true")
    analyze.add_argument(
        "--question",
        default="",
        help="optional bounded research question supplied to the WSP prompt",
    )

    apply = commands.add_parser(
        "apply-proposal", help="Apply a reviewed proposal to a separate candidate copy."
    )
    apply.add_argument("--proposal", type=Path, required=True)
    apply.add_argument("--baseline", type=Path, default=DEFAULT_MODEL)
    apply.add_argument("--dataset", type=Path, required=True)
    apply.add_argument("--candidate", type=Path, required=True)
    apply.add_argument("--approve-candidate-edit", action="store_true")

    evaluate = commands.add_parser(
        "evaluate", help="Compare candidate and baseline on development or sealed data."
    )
    evaluate.add_argument("--baseline", type=Path, default=DEFAULT_MODEL)
    evaluate.add_argument("--candidate", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--split", choices=("development", "sealed"), required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    accept = commands.add_parser(
        "accept", help="Accept a candidate only after sealed-set gates and explicit approval."
    )
    accept.add_argument("--report", type=Path, required=True)
    accept.add_argument("--output", type=Path, required=True)
    accept.add_argument("--confirm-accept", action="store_true")

    benchmark = commands.add_parser(
        "benchmark", help="Run paired playing-strength games for a candidate."
    )
    benchmark.add_argument("--baseline", type=Path, default=DEFAULT_MODEL)
    benchmark.add_argument("--candidate", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--self-play-pairs", type=int, default=2)
    benchmark.add_argument("--stockfish-games", type=int, default=0)
    benchmark.add_argument("--stockfish", default=os.environ.get("STOCKFISH_EXECUTABLE", "stockfish"))
    benchmark.add_argument("--depth", type=int, default=8)
    benchmark.add_argument("--threads", type=int, default=1)
    benchmark.add_argument("--hash-mb", type=int, default=64)
    benchmark.add_argument("--max-plies", type=int, default=300)

    args = parser.parse_args()
    if args.command == "prepare-data":
        prepare_datasets(
            args.pgn,
            args.output_dir,
            stockfish=args.stockfish,
            cache_path=args.cache,
            max_games=args.max_games,
            max_positions_per_split=args.max_positions_per_split,
            samples_per_game=args.samples_per_game,
            seed=args.seed,
            depth=args.depth,
            time_limit=args.time_limit,
            threads=args.threads,
            hash_mb=args.hash_mb,
            cp_scale=args.cp_scale,
            overwrite=args.overwrite_data,
        )
    elif args.command == "analyze":
        run_analysis(
            args.model,
            args.dataset,
            args.output_dir,
            model_id=args.model_id,
            max_steps=args.max_steps,
            allow_edit_proposal=args.propose_edit,
            question=args.question,
        )
    elif args.command == "apply-proposal":
        report = apply_proposal(
            args.proposal,
            args.baseline,
            args.dataset,
            args.candidate,
            approved=args.approve_candidate_edit,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "evaluate":
        report = evaluate_candidate(
            args.baseline,
            args.candidate,
            args.dataset,
            args.output,
            args.split,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "accept":
        report = accept_candidate(
            args.report,
            args.output,
            confirmed=args.confirm_accept,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "benchmark":
        report = benchmark_playing_strength(
            args.baseline,
            args.candidate,
            self_play_pairs=args.self_play_pairs,
            stockfish_games=args.stockfish_games,
            stockfish_path=args.stockfish,
            depth=args.depth,
            threads=args.threads,
            hash_mb=args.hash_mb,
            max_plies=args.max_plies,
        )
        _atomic_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
