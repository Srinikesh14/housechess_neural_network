"""Compare V1/V2 regression quality and optionally run playing benchmarks."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

from backend.encoder import INPUT_SIZE, LEGACY_INPUT_SIZE, encode_board, encode_board_v1
from backend.mlp import MLP
from backend.stockfish_eval import StockfishEvaluationError, StockfishEvaluator
from backend.train import (
    DEFAULT_CACHE,
    DEFAULT_PGN,
    _iter_pgn,
    _sampled_boards,
)

DEFAULT_V1 = Path(__file__).with_name("models") / "model_v1_original.npz"
DEFAULT_V2 = Path(__file__).with_name("models") / "model_v2_stockfish.npz"


def _encode(model: MLP, board: chess.Board):
    if model.W1.shape[0] == LEGACY_INPUT_SIZE:
        return encode_board_v1(board)
    if model.W1.shape[0] == INPUT_SIZE:
        return encode_board(board)
    raise ValueError(f"Unsupported model input dimension: {model.W1.shape[0]}")


@dataclass
class RegressionMetrics:
    count: int = 0
    squared_error: float = 0.0
    absolute_error: float = 0.0
    sum_prediction: float = 0.0
    sum_target: float = 0.0
    sum_prediction_squared: float = 0.0
    sum_target_squared: float = 0.0
    sum_product: float = 0.0
    runtime_seconds: float = 0.0
    max_repeat_difference: float = 0.0

    def add(self, prediction: float, target: float, elapsed: float, repeated: float) -> None:
        difference = prediction - target
        self.count += 1
        self.squared_error += difference * difference
        self.absolute_error += abs(difference)
        self.sum_prediction += prediction
        self.sum_target += target
        self.sum_prediction_squared += prediction * prediction
        self.sum_target_squared += target * target
        self.sum_product += prediction * target
        self.runtime_seconds += elapsed
        self.max_repeat_difference = max(
            self.max_repeat_difference, abs(prediction - repeated)
        )

    def report(self) -> dict[str, float | int | None]:
        if self.count == 0:
            raise ValueError("No validation positions were evaluated.")
        count = self.count
        covariance = self.sum_product - self.sum_prediction * self.sum_target / count
        prediction_variance = (
            self.sum_prediction_squared - self.sum_prediction**2 / count
        )
        target_variance = self.sum_target_squared - self.sum_target**2 / count
        denominator = max(0.0, prediction_variance) ** 0.5 * max(
            0.0, target_variance
        ) ** 0.5
        correlation = covariance / denominator if denominator > 0 else None
        return {
            "positions": count,
            "mse": self.squared_error / count,
            "mae": self.absolute_error / count,
            "correlation": correlation,
            "runtime_ms_per_position": self.runtime_seconds * 1000.0 / count,
            "max_repeat_difference": self.max_repeat_difference,
        }


def evaluate_regression(
    pgn_path: Path,
    v1: MLP,
    v2: MLP,
    teacher: StockfishEvaluator,
    *,
    max_games: int | None = None,
    max_positions: int | None = None,
    samples_per_game: int = 4,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate both models on identical sampled positions from held-out games."""
    metrics = {"v1": RegressionMetrics(), "v2": RegressionMetrics()}
    candidate_positions = 0
    skipped_positions = 0
    cache_hits_before = teacher.cache_hits
    new_evaluations_before = teacher.new_evaluations
    for game_number, game in _iter_pgn(pgn_path):
        if max_games is not None and game_number > max_games:
            break
        if max_positions is not None and candidate_positions >= max_positions:
            break
        if game_number % 5 != 0 or game.errors:
            continue
        try:
            boards = _sampled_boards(game, game_number, samples_per_game, seed + 1_000_003)
        except (ValueError, chess.IllegalMoveError):
            continue
        for board in boards:
            if max_positions is not None and candidate_positions >= max_positions:
                break
            candidate_positions += 1
            try:
                target = teacher.evaluate(board).target
            except StockfishEvaluationError as error:
                skipped_positions += 1
                if skipped_positions == 1:
                    print(f"Skipping failed Stockfish evaluation: {error}", flush=True)
                continue
            for name, model in (("v1", v1), ("v2", v2)):
                features = _encode(model, board)
                started = time.perf_counter()
                prediction = model.forward(features)
                elapsed = time.perf_counter() - started
                repeated_prediction = model.forward(features)
                metrics[name].add(prediction, target, elapsed, repeated_prediction)
            if max_positions is not None and candidate_positions >= max_positions:
                break
    return {
        "v1": metrics["v1"].report(),
        "v2": metrics["v2"].report(),
        "positions_attempted": candidate_positions,
        "skipped_positions": skipped_positions,
        "cache_hits": teacher.cache_hits - cache_hits_before,
        "new_stockfish_evaluations": teacher.new_evaluations - new_evaluations_before,
    }


def _choose_model_move(model: MLP, board: chess.Board) -> chess.Move:
    moving_color = board.turn
    best_move: chess.Move | None = None
    best_score: float | None = None
    for move in list(board.legal_moves):
        board.push(move)
        white_score = model.forward(_encode(model, board))
        board.pop()
        moving_score = white_score if moving_color == chess.WHITE else -white_score
        if best_score is None or moving_score > best_score:
            best_move, best_score = move, moving_score
    if best_move is None:
        raise ValueError("Cannot select a move from a position with no legal moves.")
    return best_move


@dataclass
class GameResults:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_plies: int = 0

    def add(self, board: chess.Board, model_color: chess.Color, plies: int) -> None:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            self.draws += 1
        elif outcome.winner == model_color:
            self.wins += 1
        else:
            self.losses += 1
        self.total_plies += plies

    def report(self) -> dict[str, float | int]:
        games = self.wins + self.losses + self.draws
        return {
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": (self.wins + 0.5 * self.draws) / games if games else 0.0,
            "average_game_length_plies": self.total_plies / games if games else 0.0,
            "games": games,
        }


def _play_model_game(
    model: MLP,
    model_color: chess.Color,
    engine: chess.engine.SimpleEngine | None = None,
    engine_depth: int = 8,
    max_plies: int = 300,
) -> tuple[chess.Board, int]:
    board = chess.Board()
    plies = 0
    while plies < max_plies and not board.is_game_over(claim_draw=True):
        if board.turn == model_color:
            move = _choose_model_move(model, board)
        elif engine is not None:
            move = engine.play(board, chess.engine.Limit(depth=engine_depth)).move
            if move is None:
                raise RuntimeError("Stockfish returned no move in a non-terminal position.")
        else:
            raise ValueError("An opponent engine is required for this game.")
        board.push(move)
        plies += 1
    return board, plies


def play_benchmarks(
    v1: MLP,
    v2: MLP,
    *,
    self_play_pairs: int = 0,
    stockfish_games: int = 0,
    stockfish_path: str = "stockfish",
    depth: int = 8,
    threads: int = 1,
    hash_mb: int = 64,
    max_plies: int = 300,
) -> dict[str, dict[str, float | int]]:
    """Play paired same-start/color games; report outcomes from each model's view."""
    results: dict[str, dict[str, float | int]] = {}
    if self_play_pairs:
        v1_results = GameResults()
        v2_results = GameResults()
        for _ in range(self_play_pairs):
            board, plies = _play_models(v1, v2, chess.WHITE, max_plies)
            v1_results.add(board, chess.WHITE, plies)
            v2_results.add(board, chess.BLACK, plies)
            board, plies = _play_models(v2, v1, chess.WHITE, max_plies)
            v2_results.add(board, chess.WHITE, plies)
            v1_results.add(board, chess.BLACK, plies)
        results["v1_vs_v2"] = v1_results.report()
        results["v2_vs_v1"] = v2_results.report()

    if stockfish_games:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        try:
            engine.configure({"Threads": threads, "Hash": hash_mb})
            for name, model in (("v1", v1), ("v2", v2)):
                model_results = GameResults()
                for game_index in range(stockfish_games):
                    model_color = chess.WHITE if game_index % 2 == 0 else chess.BLACK
                    board, plies = _play_model_game(
                        model, model_color, engine, depth, max_plies
                    )
                    model_results.add(board, model_color, plies)
                results[f"{name}_vs_stockfish"] = model_results.report()
        finally:
            engine.quit()
    return results


def _play_models(
    white_model: MLP,
    black_model: MLP,
    white_color: chess.Color,
    max_plies: int,
) -> tuple[chess.Board, int]:
    board = chess.Board()
    plies = 0
    while plies < max_plies and not board.is_game_over(claim_draw=True):
        model = white_model if board.turn == white_color else black_model
        board.push(_choose_model_move(model, board))
        plies += 1
    return board, plies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", type=Path, default=DEFAULT_PGN)
    parser.add_argument("--v1", type=Path, default=DEFAULT_V1)
    parser.add_argument("--v2", type=Path, default=DEFAULT_V2)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--cp-scale", type=float, default=400.0)
    parser.add_argument("--max-games", type=int, default=1000)
    parser.add_argument("--max-positions", type=int, default=1000)
    parser.add_argument("--samples-per-game", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-play-pairs", type=int, default=0)
    parser.add_argument("--stockfish-games", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=300)
    args = parser.parse_args()

    v1, v2 = MLP.load(args.v1), MLP.load(args.v2)
    if v1.W1.shape[0] != LEGACY_INPUT_SIZE or v2.W1.shape[0] != INPUT_SIZE:
        raise ValueError("Expected V1=773 inputs and V2=838 inputs.")
    report: dict[str, object] = {}
    with StockfishEvaluator(
        args.stockfish,
        args.cache,
        depth=args.depth,
        time_limit=args.time_limit,
        threads=args.threads,
        hash_mb=args.hash_mb,
        cp_scale=args.cp_scale,
    ) as teacher:
        report["regression"] = evaluate_regression(
            args.pgn,
            v1,
            v2,
            teacher,
            max_games=args.max_games,
            max_positions=args.max_positions,
            samples_per_game=args.samples_per_game,
            seed=args.seed,
        )
    if args.self_play_pairs or args.stockfish_games:
        report["playing"] = play_benchmarks(
            v1,
            v2,
            self_play_pairs=args.self_play_pairs,
            stockfish_games=args.stockfish_games,
            stockfish_path=args.stockfish,
            depth=args.depth,
            threads=args.threads,
            hash_mb=args.hash_mb,
            max_plies=args.max_plies,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
