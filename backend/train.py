"""Train the NumPy evaluator from PGN game results."""

from __future__ import annotations

import argparse
import io
import os
import random
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

import chess
import chess.pgn
import numpy as np
import zstandard

from backend.encoder import encode_board
from backend.mlp import MLP

DEFAULT_PGN = Path(__file__).resolve().parents[2] / "lichess_db_standard_rated_2016-11.pgn.zst"


@contextmanager
def open_pgn(path: Path) -> Iterator[TextIO]:
    """Open plain-text PGN or stream a .zst-compressed PGN."""
    with path.open("rb") as binary:
        if path.suffix.lower() == ".zst":
            reader = zstandard.ZstdDecompressor().stream_reader(binary)
            text = io.TextIOWrapper(reader, encoding="utf-8")
            try:
                yield text
            finally:
                text.close()
        else:
            text = io.TextIOWrapper(binary, encoding="utf-8")
            try:
                yield text
            finally:
                text.close()


def train_epoch(
    model: MLP,
    pgn_path: Path,
    epoch: int,
    max_games: int | None,
    max_positions: int | None,
    samples_per_game: int,
    learning_rate: float,
    seed: int,
    start_game: int,
    checkpoint_path: Path,
    checkpoint_interval: int,
    progress_interval: int,
) -> tuple[int, int, int, float, int, float, bool]:
    """Train and validate one archive pass without keeping positions in memory."""
    parsed_games = 0
    labeled_games = 0
    processed_positions = 0
    training_loss = 0.0
    training_count = 0
    validation_loss = 0.0
    validation_count = 0
    reached_eof = False
    started = time.monotonic()

    with open_pgn(pgn_path) as pgn:
        while max_games is None or parsed_games < max_games:
            if max_positions is not None and processed_positions >= max_positions:
                break
            game = chess.pgn.read_game(pgn)
            if game is None:
                reached_eof = True
                break
            parsed_games += 1
            if parsed_games <= start_game:
                continue

            result = game.headers.get("Result")
            if result in ("1-0", "0-1", "1/2-1/2"):
                moves = list(game.mainline_moves())
                position_count = len(moves) + 1
                sample_count = min(samples_per_game, position_count)
                game_rng = random.Random(seed + parsed_games)
                sampled_plies = sorted(game_rng.sample(range(position_count), sample_count))
                target = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}[result]
                is_validation_game = parsed_games % 5 == 0
                board = game.board()
                sample_index = 0

                for ply in range(position_count):
                    if sample_index == sample_count:
                        break
                    if ply == sampled_plies[sample_index]:
                        features = encode_board(board)
                        if is_validation_game:
                            difference = model.forward(features) - target
                            validation_loss += difference * difference
                            validation_count += 1
                        else:
                            training_loss += model.train_step(features, target, learning_rate)
                            training_count += 1
                        processed_positions += 1
                        sample_index += 1
                    if ply < len(moves):
                        board.push(moves[ply])

                if sample_index:
                    labeled_games += 1

            if parsed_games % checkpoint_interval == 0:
                save_checkpoint(model, checkpoint_path, epoch, parsed_games, completed=False)
            if parsed_games % progress_interval == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"Epoch {epoch}: {parsed_games:,} games parsed, "
                    f"{processed_positions:,} sampled positions processed "
                    f"({processed_positions / elapsed:.1f} positions/s); "
                    f"checkpoint saved at {checkpoint_path}.",
                    flush=True,
                )

    return (
        parsed_games,
        labeled_games,
        training_count,
        training_loss,
        validation_count,
        validation_loss,
        reached_eof,
    )


def save_checkpoint(
    model: MLP,
    path: Path,
    epoch: int,
    games_processed: int,
    completed: bool,
) -> None:
    """Atomically save model weights and enough progress to resume a pass."""
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
                epoch=np.asarray(epoch, dtype=np.int64),
                games_processed=np.asarray(games_processed, dtype=np.int64),
                completed=np.asarray(completed, dtype=np.bool_),
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(path: Path) -> tuple[MLP, int, int, bool]:
    """Load a checkpoint's model weights and archive progress."""
    model = MLP.load(path)
    with np.load(path, allow_pickle=False) as archive:
        required = ("epoch", "games_processed", "completed")
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"Checkpoint is missing progress fields: {', '.join(missing)}")
        epoch = int(archive["epoch"])
        games_processed = int(archive["games_processed"])
        completed = bool(archive["completed"])
    if epoch <= 0 or games_processed < 0:
        raise ValueError("Checkpoint progress values are invalid.")
    return model, epoch, games_processed, completed


def train(
    pgn_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    max_games: int | None = None,
    max_positions: int | None = None,
    samples_per_game: int = 4,
    epochs: int = 1,
    learning_rate: float = 0.001,
    seed: int = 42,
    resume: bool = False,
    checkpoint_interval: int = 10000,
    progress_interval: int = 50000,
) -> MLP:
    if not pgn_path.is_file():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")
    if any(value is not None and value <= 0 for value in (max_games, max_positions)):
        raise ValueError("Game and position limits must be positive when provided.")
    if min(samples_per_game, epochs) <= 0:
        raise ValueError("Samples per game and epochs must be positive.")
    if min(checkpoint_interval, progress_interval) <= 0:
        raise ValueError("Checkpoint and progress intervals must be positive.")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("Learning rate must be a positive finite number.")

    start_epoch = 1
    start_game = 0
    if resume and checkpoint_path.is_file():
        model, start_epoch, start_game, completed = load_checkpoint(checkpoint_path)
        if completed:
            raise ValueError(
                f"Checkpoint {checkpoint_path} already marks training complete. "
                "Choose a new checkpoint path to start another run."
            )
        print(
            f"Resuming from epoch {start_epoch}, after game {start_game:,}; "
            f"loaded weights from {checkpoint_path}.",
            flush=True,
        )
    elif resume and output_path.is_file():
        model = MLP.load(output_path)
        print(
            f"No progress checkpoint exists; starting the archive from game 1 "
            f"using the existing model weights in {output_path}.",
            flush=True,
        )
    else:
        model = MLP(seed=seed)

    print(
        f"Training from {pgn_path.resolve()} for {epochs} epoch(s); "
        f"sampling up to {samples_per_game} positions per game; "
        f"checkpoint interval={checkpoint_interval:,} games, "
        f"progress interval={progress_interval:,} games.",
        flush=True,
    )
    parsed = start_game
    completed_all_epochs = True
    for epoch in range(start_epoch, epochs + 1):
        (
            parsed,
            labeled,
            train_count,
            train_loss,
            validation_count,
            validation_loss,
            reached_eof,
        ) = train_epoch(
            model,
            pgn_path,
            epoch,
            max_games,
            max_positions,
            samples_per_game,
            learning_rate,
            seed + epoch * 1_000_003,
            start_game if epoch == start_epoch else 0,
            checkpoint_path,
            checkpoint_interval,
            progress_interval,
        )
        if train_count == 0 or validation_count == 0:
            raise ValueError(
                "The archive pass did not produce both training and validation positions. "
                "Use a larger game/position limit or a different PGN."
            )
        average_training_loss = train_loss / train_count
        average_validation_loss = validation_loss / validation_count
        print(
            f"Epoch {epoch}/{epochs}: read {parsed:,} games ({labeled:,} with results), "
            f"{train_count:,} training positions, {validation_count:,} validation positions; "
            f"training MSE={average_training_loss:.5f}, "
            f"validation MSE={average_validation_loss:.5f}",
            flush=True,
        )
        if max_games is None and max_positions is None and reached_eof:
            save_checkpoint(model, checkpoint_path, epoch + 1, 0, completed=False)
        else:
            save_checkpoint(model, checkpoint_path, epoch, parsed, completed=False)
            completed_all_epochs = False
        start_game = 0

    model.save(output_path)
    if completed_all_epochs:
        save_checkpoint(model, checkpoint_path, epochs, parsed, completed=True)
    print(f"Saved trained model to {output_path.resolve()}")
    if completed_all_epochs:
        print(f"Saved completed checkpoint to {checkpoint_path.resolve()}")
    else:
        print(f"Saved resumable checkpoint to {checkpoint_path.resolve()}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", type=Path, default=DEFAULT_PGN, help="PGN or .pgn.zst file")
    parser.add_argument("--output", type=Path, default=Path("backend/model.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("backend/training_checkpoint.npz"))
    parser.add_argument(
        "--max-games", type=int, default=None, help="optional game limit (default: entire archive)"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="optional sampled-position limit (default: entire archive)",
    )
    parser.add_argument("--samples-per-game", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    parser.add_argument("--progress-interval", type=int, default=50000)
    arguments = parser.parse_args()
    train(
        arguments.pgn,
        arguments.output,
        arguments.checkpoint,
        max_games=arguments.max_games,
        max_positions=arguments.max_positions,
        samples_per_game=arguments.samples_per_game,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
        resume=arguments.resume,
        checkpoint_interval=arguments.checkpoint_interval,
        progress_interval=arguments.progress_interval,
    )


if __name__ == "__main__":
    main()
