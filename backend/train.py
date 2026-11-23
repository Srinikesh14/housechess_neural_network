"""Train the V2 NumPy evaluator from Stockfish-labeled PGN positions."""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO

import chess
import chess.pgn
import numpy as np
import zstandard

from backend.encoder import INPUT_SIZE, encode_board
from backend.mlp import MLP
from backend.stockfish_eval import StockfishEvaluationError, StockfishEvaluator

DEFAULT_PGN = Path(__file__).resolve().parents[2] / "lichess_db_standard_rated_2016-11.pgn.zst"
DEFAULT_OUTPUT = Path(__file__).with_name("models") / "model_v2_stockfish.npz"
DEFAULT_CHECKPOINT = Path(__file__).with_name("models") / "training_checkpoint_v2.npz"
DEFAULT_CACHE = Path(__file__).with_name("models") / "stockfish_cache.sqlite3"


@contextmanager
def open_pgn(path: Path) -> Iterator[TextIO]:
    """Open plain-text PGN or stream a .zst-compressed PGN."""
    with path.open("rb") as binary:
        if path.suffix.lower() == ".zst":
            reader = zstandard.ZstdDecompressor().stream_reader(binary)
            text = io.TextIOWrapper(reader, encoding="utf-8")
        else:
            text = io.TextIOWrapper(binary, encoding="utf-8")
        try:
            yield text
        finally:
            text.close()


def _sampled_boards(
    game: chess.pgn.Game, game_number: int, samples_per_game: int, seed: int
) -> list[chess.Board]:
    moves = list(game.mainline_moves())
    position_count = len(moves) + 1
    sample_count = min(samples_per_game, position_count)
    sample_rng = random.Random(seed + game_number)
    sampled_plies = set(sample_rng.sample(range(position_count), sample_count))
    board = game.board()
    sampled: list[chess.Board] = []
    for ply in range(position_count):
        if ply in sampled_plies:
            sampled.append(board.copy(stack=True))
        if ply < len(moves):
            board.push(moves[ply])
    return sampled


def _configuration_signature(
    pgn_path: Path,
    samples_per_game: int,
    learning_rate: float,
    seed: int,
    depth: int | None,
    time_limit: float | None,
    threads: int,
    hash_mb: int,
    cp_scale: float,
    engine_signature: str | None,
) -> str:
    stat = pgn_path.stat()
    configuration = {
        "pgn": str(pgn_path.resolve()),
        "pgn_size": stat.st_size,
        "pgn_mtime_ns": stat.st_mtime_ns,
        "input_size": INPUT_SIZE,
        "samples_per_game": samples_per_game,
        "learning_rate": learning_rate,
        "seed": seed,
        "depth": depth,
        "time_limit": time_limit,
        "threads": threads,
        "hash_mb": hash_mb,
        "cp_scale": cp_scale,
        "engine_signature": engine_signature,
        "validation_game_modulus": 5,
    }
    return json.dumps(configuration, sort_keys=True, separators=(",", ":"))


def save_checkpoint(
    model: MLP,
    path: Path,
    *,
    epoch: int,
    games_processed: int,
    positions_processed: int,
    training_loss_sum: float,
    training_count: int,
    completed: bool,
    configuration: str,
) -> None:
    """Atomically save V2 parameters and resumable progress."""
    if model.W1.shape[0] != INPUT_SIZE:
        raise ValueError("A V2 checkpoint must contain an 838-input model.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", prefix=f".{path.stem}-", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                W1=model.W1,
                b1=model.b1,
                W2=model.W2,
                b2=model.b2,
                W3=model.W3,
                b3=model.b3,
                model_version=np.asarray(2, dtype=np.int64),
                epoch=np.asarray(epoch, dtype=np.int64),
                games_processed=np.asarray(games_processed, dtype=np.int64),
                positions_processed=np.asarray(positions_processed, dtype=np.int64),
                training_loss_sum=np.asarray(training_loss_sum, dtype=np.float64),
                training_count=np.asarray(training_count, dtype=np.int64),
                completed=np.asarray(completed, dtype=np.bool_),
                configuration=np.asarray(configuration),
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(path: Path) -> tuple[MLP, dict[str, Any]]:
    """Load a V2 checkpoint and reject legacy or incomplete schemas."""
    with np.load(path, allow_pickle=False) as archive:
        input_size = int(archive["W1"].shape[0]) if "W1" in archive else 0
        if input_size != INPUT_SIZE or (
            "model_version" in archive and int(archive["model_version"]) != 2
        ):
            raise ValueError(
                f"Checkpoint {path} is not a V2 838-input checkpoint; "
                "the legacy V1 checkpoint cannot resume V2 training."
            )
        required = (
            "epoch",
            "games_processed",
            "positions_processed",
            "training_loss_sum",
            "training_count",
            "completed",
            "configuration",
        )
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"Checkpoint is missing progress fields: {', '.join(missing)}")
        state = {
            "epoch": int(archive["epoch"]),
            "games_processed": int(archive["games_processed"]),
            "positions_processed": int(archive["positions_processed"]),
            "training_loss_sum": float(archive["training_loss_sum"]),
            "training_count": int(archive["training_count"]),
            "completed": bool(archive["completed"]),
            "configuration": str(archive["configuration"]),
        }
    if (
        state["epoch"] <= 0
        or state["games_processed"] < 0
        or state["positions_processed"] < 0
        or state["training_count"] < 0
    ):
        raise ValueError("Checkpoint progress values are invalid.")
    return MLP.load(path), state


def _iter_pgn(path: Path) -> Iterator[tuple[int, chess.pgn.Game]]:
    with open_pgn(path) as pgn:
        game_number = 0
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                return
            game_number += 1
            yield game_number, game


def train(
    pgn_path: Path,
    output_path: Path = DEFAULT_OUTPUT,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    *,
    cache_path: Path = DEFAULT_CACHE,
    stockfish: str | Path = "stockfish",
    max_games: int | None = None,
    max_positions: int | None = None,
    samples_per_game: int = 4,
    epochs: int = 1,
    learning_rate: float = 0.001,
    seed: int = 42,
    depth: int | None = 8,
    time_limit: float | None = None,
    threads: int = 1,
    hash_mb: int = 64,
    cp_scale: float = 400.0,
    resume: bool = False,
    overwrite: bool = False,
    checkpoint_interval: int = 1000,
    progress_interval: int = 5000,
    evaluator: Any | None = None,
) -> MLP:
    """Train streaming PGN positions; every fifth parsed game is held out."""
    pgn_path = Path(pgn_path)
    output_path = Path(output_path)
    checkpoint_path = Path(checkpoint_path)
    cache_path = Path(cache_path)
    if not pgn_path.is_file():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")
    if any(value is not None and value <= 0 for value in (max_games, max_positions)):
        raise ValueError("Game and position limits must be positive when provided.")
    if min(samples_per_game, epochs, checkpoint_interval, progress_interval) <= 0:
        raise ValueError("Samples, epochs, and intervals must be positive.")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("Learning rate must be positive and finite.")
    if depth is None and time_limit is None:
        raise ValueError("Provide a Stockfish depth or time limit.")
    if depth is not None and depth <= 0:
        raise ValueError("Stockfish depth must be positive.")
    if time_limit is not None and (not np.isfinite(time_limit) or time_limit <= 0.0):
        raise ValueError("Stockfish time limit must be positive and finite.")
    if threads <= 0 or hash_mb <= 0:
        raise ValueError("Stockfish threads and hash memory must be positive.")
    if not np.isfinite(cp_scale) or cp_scale <= 0.0:
        raise ValueError("Centipawn scale must be positive and finite.")
    if resume and overwrite:
        raise ValueError("Choose either --resume or --overwrite, not both.")
    protected_paths = {
        Path(__file__).with_name("model.npz").resolve(),
        Path(__file__).with_name("training_checkpoint.npz").resolve(),
        (Path(__file__).with_name("models") / "model_v1_original.npz").resolve(),
        (Path(__file__).with_name("models") / "training_checkpoint_v1_original.npz").resolve(),
    }
    if output_path.resolve() in protected_paths or checkpoint_path.resolve() in protected_paths:
        raise ValueError("V2 training must not overwrite V1 model or checkpoint files.")
    if output_path.resolve() == checkpoint_path.resolve():
        raise ValueError("Model output and checkpoint paths must be different.")
    if not resume and not overwrite and (output_path.exists() or checkpoint_path.exists()):
        raise FileExistsError(
            "V2 output/checkpoint already exists. Use --resume to continue or "
            "--overwrite with new output paths to start over."
        )

    teacher = evaluator or StockfishEvaluator(
        stockfish,
        cache_path,
        depth=depth,
        time_limit=time_limit,
        threads=threads,
        hash_mb=hash_mb,
        cp_scale=cp_scale,
    )
    start_epoch = 1
    start_game = 0
    positions_processed = 0
    training_loss_sum = 0.0
    training_count = 0
    completed = False

    with teacher:
        configuration = _configuration_signature(
            pgn_path,
            samples_per_game,
            learning_rate,
            seed,
            depth,
            time_limit,
            threads,
            hash_mb,
            cp_scale,
            getattr(teacher, "engine_signature", None),
        )
        if resume and checkpoint_path.is_file():
            model, state = load_checkpoint(checkpoint_path)
            if state["completed"]:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} is complete. Choose a new path "
                    "or increase the epoch count using a new checkpoint."
                )
            if state["configuration"] != configuration:
                raise ValueError(
                    "Checkpoint training configuration differs from this run. "
                    "Keep the PGN, Stockfish settings, seed, sample rate, and learning "
                    "rate unchanged when resuming."
                )
            start_epoch = state["epoch"]
            if start_epoch > epochs:
                raise ValueError(
                    f"Checkpoint is at epoch {start_epoch}; set --epochs to at least "
                    f"{start_epoch} to resume."
                )
            start_game = state["games_processed"]
            positions_processed = state["positions_processed"]
            training_loss_sum = state["training_loss_sum"]
            training_count = state["training_count"]
            print(
                f"Resuming V2 epoch {start_epoch} after game {start_game:,} "
                f"({positions_processed:,} sampled positions).",
                flush=True,
            )
        elif resume and output_path.is_file():
            model = MLP.load(output_path)
            if model.W1.shape[0] != INPUT_SIZE:
                raise ValueError("The selected model is not a V2 838-input model.")
            print("No V2 checkpoint exists; scanning from the start with V2 weights.", flush=True)
        else:
            model = MLP(seed=seed)

        print(
            f"V2 training: {pgn_path.resolve()}, Stockfish depth={depth}, "
            f"time_limit={time_limit}, threads={threads}, hash={hash_mb} MB, "
            f"K={cp_scale:g}; {samples_per_game} sampled position(s) per game.",
            flush=True,
        )

        for epoch in range(start_epoch, epochs + 1):
            epoch_started = time.monotonic()
            train_loss_sum = training_loss_sum if epoch == start_epoch else 0.0
            train_count = training_count if epoch == start_epoch else 0
            skipped_positions = 0
            invalid_games = 0
            last_game = start_game
            hits_before = getattr(teacher, "cache_hits", 0)
            new_before = getattr(teacher, "new_evaluations", 0)
            pass_reached_eof = False

            for game_number, game in _iter_pgn(pgn_path):
                if game_number <= start_game:
                    continue
                if max_games is not None and game_number > max_games:
                    break
                if max_positions is not None and positions_processed >= max_positions:
                    break
                if game.errors:
                    invalid_games += 1
                    last_game = game_number
                    continue
                try:
                    boards = _sampled_boards(
                        game,
                        game_number,
                        samples_per_game,
                        seed + epoch * 1_000_003,
                    )
                except (ValueError, chess.IllegalMoveError):
                    invalid_games += 1
                    last_game = game_number
                    continue

                is_validation_game = game_number % 5 == 0
                for board in boards:
                    positions_processed += 1
                    try:
                        target = teacher.evaluate(board).target
                    except StockfishEvaluationError as error:
                        skipped_positions += 1
                        if skipped_positions == 1:
                            print(f"Skipping failed Stockfish evaluation: {error}", flush=True)
                        continue
                    if not is_validation_game:
                        loss = model.train_step(encode_board(board), target, learning_rate)
                        train_loss_sum += loss
                        train_count += 1

                last_game = game_number
                if game_number % checkpoint_interval == 0:
                    save_checkpoint(
                        model,
                        checkpoint_path,
                        epoch=epoch,
                        games_processed=last_game,
                        positions_processed=positions_processed,
                        training_loss_sum=train_loss_sum,
                        training_count=train_count,
                        completed=False,
                        configuration=configuration,
                    )
                if game_number % progress_interval == 0:
                    elapsed = max(time.monotonic() - epoch_started, 1e-9)
                    print(
                        f"Epoch {epoch}: {game_number:,} games; "
                        f"{positions_processed:,} positions; "
                        f"{getattr(teacher, 'cache_hits', 0) - hits_before:,} cache hits; "
                        f"{getattr(teacher, 'new_evaluations', 0) - new_before:,} new "
                        f"Stockfish evaluations; {elapsed:.1f}s elapsed.",
                        flush=True,
                    )

            else:
                pass_reached_eof = True

            # Determine whether the selected archive pass was truncated by limits.
            limited = (
                (max_games is not None and last_game >= max_games)
                or (max_positions is not None and positions_processed >= max_positions)
            )
            if not pass_reached_eof and not limited and max_games is None:
                pass_reached_eof = True
            if train_count == 0:
                raise ValueError(
                    "This pass produced no training positions. Increase the game/position "
                    "limit or adjust the whole-game validation split."
                )

            validation_sum = 0.0
            validation_count = 0
            replayed_positions = 0
            for game_number, game in _iter_pgn(pgn_path):
                if game_number > last_game or replayed_positions >= positions_processed:
                    break
                if game.errors:
                    continue
                if game_number % 5 != 0:
                    replayed_positions += min(
                        samples_per_game, len(list(game.mainline_moves())) + 1
                    )
                    continue
                try:
                    boards = _sampled_boards(
                        game,
                        game_number,
                        samples_per_game,
                        seed + epoch * 1_000_003,
                    )
                except (ValueError, chess.IllegalMoveError):
                    continue
                for board in boards:
                    if replayed_positions >= positions_processed:
                        break
                    replayed_positions += 1
                    try:
                        target = teacher.evaluate(board).target
                    except StockfishEvaluationError:
                        continue
                    difference = model.forward(encode_board(board)) - target
                    validation_sum += difference * difference
                    validation_count += 1

            if train_count == 0:
                raise ValueError("No Stockfish-labeled training positions were processed.")
            if validation_count == 0:
                raise ValueError(
                    "No Stockfish-labeled validation positions were produced. "
                    "Use at least five valid PGN games and a larger position limit."
                )

            training_mse = train_loss_sum / train_count
            validation_mse = validation_sum / validation_count
            print(
                f"Epoch {epoch}: {positions_processed:,} sampled/evaluated attempts; "
                f"training MSE={training_mse:.6f} ({train_count:,} positions), "
                f"validation MSE={validation_mse:.6f} ({validation_count:,} positions); "
                f"{invalid_games:,} invalid games, {skipped_positions:,} skipped positions, "
                f"{getattr(teacher, 'cache_hits', 0) - hits_before:,} cache hits, "
                f"{getattr(teacher, 'new_evaluations', 0) - new_before:,} new Stockfish "
                f"evaluations; {time.monotonic() - epoch_started:.1f}s.",
                flush=True,
            )

            model.save(output_path)
            if pass_reached_eof and not limited:
                next_epoch = epoch + 1
                training_loss_sum = 0.0
                training_count = 0
                start_game = 0
                if epoch == epochs:
                    completed = True
                    next_epoch = epoch
                save_checkpoint(
                    model,
                    checkpoint_path,
                    epoch=next_epoch,
                    games_processed=0,
                    positions_processed=0,
                    training_loss_sum=0.0,
                    training_count=0,
                    completed=completed,
                    configuration=configuration,
                )
            else:
                save_checkpoint(
                    model,
                    checkpoint_path,
                    epoch=epoch,
                    games_processed=last_game,
                    positions_processed=positions_processed,
                    training_loss_sum=train_loss_sum,
                    training_count=train_count,
                    completed=False,
                    configuration=configuration,
                )
                break

            positions_processed = 0
            start_game = 0

        print(f"Saved V2 model to {output_path.resolve()}")
        print(f"Saved V2 checkpoint to {checkpoint_path.resolve()}")
        print(f"Evaluation cache: {cache_path.resolve()}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", type=Path, default=DEFAULT_PGN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--stockfish",
        default=os.environ.get("STOCKFISH_EXECUTABLE", "stockfish"),
        help="Stockfish executable path (or STOCKFISH_EXECUTABLE environment variable)",
    )
    parser.add_argument("--depth", type=int, default=8, help="Stockfish search depth")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="optional per-position search time in seconds; may be combined with --depth",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--cp-scale", type=float, default=400.0, help="K for tanh(cp / K)")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--samples-per-game", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing the selected V2 model/checkpoint (never touches V1 files)",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--progress-interval", type=int, default=5000)
    args = parser.parse_args()
    train(
        args.pgn,
        args.output,
        args.checkpoint,
        cache_path=args.cache,
        stockfish=args.stockfish,
        max_games=args.max_games,
        max_positions=args.max_positions,
        samples_per_game=args.samples_per_game,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        depth=args.depth,
        time_limit=args.time_limit,
        threads=args.threads,
        hash_mb=args.hash_mb,
        cp_scale=args.cp_scale,
        resume=args.resume,
        overwrite=args.overwrite,
        checkpoint_interval=args.checkpoint_interval,
        progress_interval=args.progress_interval,
    )


if __name__ == "__main__":
    main()
