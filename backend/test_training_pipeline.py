"""Focused tests for the V2 encoder, Stockfish cache, and training checkpoints."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np

from backend import main as api
from backend.encoder import (
    EN_PASSANT_OFFSET,
    HALFMOVE_CLOCK_OFFSET,
    INPUT_SIZE,
    LEGACY_INPUT_SIZE,
    encode_board,
    encode_board_v1,
)
from backend.evaluate import evaluate_regression, play_benchmarks
from backend.mlp import MLP
from backend.stockfish_eval import (
    PositionEvaluation,
    StockfishEvaluator,
    normalize_mate,
)
from backend.train import load_checkpoint, save_checkpoint, train


class FakeEngine:
    id = {"name": "test-engine", "author": "unit-test"}

    def __init__(self, score: chess.engine.PovScore) -> None:
        self.score = score
        self.calls = 0

    def configure(self, _options: dict[str, int]) -> None:
        pass

    def analyse(self, _board: chess.Board, _limit: chess.engine.Limit) -> dict[str, object]:
        self.calls += 1
        return {"score": self.score}


class FakeTeacher:
    engine_signature = "synthetic-test-teacher"

    def __init__(self) -> None:
        self.cache_hits = 0
        self.new_evaluations = 0

    def __enter__(self) -> FakeTeacher:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def evaluate(self, board: chess.Board) -> PositionEvaluation:
        self.new_evaluations += 1
        score = 80 if board.turn == chess.WHITE else -80
        return PositionEvaluation("centipawn", float(score), float(np.tanh(score / 400)))


class TrainingPipelineTests(unittest.TestCase):
    def test_encoder_dimensions_en_passant_and_halfmove_clock(self) -> None:
        board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR "
            "w KQkq e6 37 2"
        )
        features = encode_board(board)
        self.assertEqual(features.shape, (INPUT_SIZE,))
        self.assertEqual(encode_board_v1(board).shape, (LEGACY_INPUT_SIZE,))
        self.assertEqual(features[EN_PASSANT_OFFSET + chess.E6], 1.0)
        self.assertEqual(np.count_nonzero(features[EN_PASSANT_OFFSET : EN_PASSANT_OFFSET + 64]), 1)
        self.assertAlmostEqual(float(features[HALFMOVE_CLOCK_OFFSET]), 0.37)

        board.halfmove_clock = 150
        self.assertEqual(encode_board(board)[HALFMOVE_CLOCK_OFFSET], 1.0)
        board.ep_square = None
        self.assertEqual(
            np.count_nonzero(encode_board(board)[EN_PASSANT_OFFSET : EN_PASSANT_OFFSET + 64]),
            0,
        )

    def test_stockfish_scores_are_converted_to_white_perspective_and_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_path = Path(temporary) / "scores.sqlite3"
            engine = FakeEngine(chess.engine.PovScore(chess.engine.Cp(200), chess.BLACK))
            board = chess.Board()
            with StockfishEvaluator(
                "fake-stockfish", cache_path, depth=4, cp_scale=400, engine=engine
            ) as teacher:
                first = teacher.evaluate(board)
                second = teacher.evaluate(board)
                self.assertEqual(first.evaluation_type, "centipawn")
                self.assertEqual(first.score, -200)
                self.assertAlmostEqual(first.target, np.tanh(-0.5))
                self.assertEqual(first, second)
                self.assertEqual(teacher.cache_hits, 1)
                self.assertEqual(teacher.new_evaluations, 1)
            self.assertEqual(engine.calls, 1)

            incompatible_engine = FakeEngine(
                chess.engine.PovScore(chess.engine.Cp(200), chess.BLACK)
            )
            with StockfishEvaluator(
                "fake-stockfish", cache_path, depth=5, cp_scale=400,
                engine=incompatible_engine
            ) as teacher:
                teacher.evaluate(board)
                self.assertEqual(teacher.cache_hits, 0)
                self.assertEqual(teacher.new_evaluations, 1)
            self.assertEqual(incompatible_engine.calls, 1)

            rescaled_engine = FakeEngine(
                chess.engine.PovScore(chess.engine.Cp(200), chess.BLACK)
            )
            with StockfishEvaluator(
                "fake-stockfish", cache_path, depth=4, cp_scale=800,
                engine=rescaled_engine
            ) as teacher:
                rescaled = teacher.evaluate(board)
                self.assertEqual(teacher.cache_hits, 1)
                self.assertAlmostEqual(rescaled.target, np.tanh(-0.25))
            self.assertEqual(rescaled_engine.calls, 0)

    def test_mate_scores_use_bounded_mate_aware_targets(self) -> None:
        self.assertGreater(normalize_mate(1), normalize_mate(12))
        self.assertGreater(normalize_mate(12), 0.9)
        self.assertLessEqual(normalize_mate(1), 1.0)
        self.assertAlmostEqual(normalize_mate(-5), -normalize_mate(5))
        with tempfile.TemporaryDirectory() as temporary:
            engine = FakeEngine(chess.engine.PovScore(chess.engine.Mate(4), chess.BLACK))
            with StockfishEvaluator(
                "fake-stockfish", Path(temporary) / "cache.sqlite3",
                depth=4, engine=engine
            ) as teacher:
                result = teacher.evaluate(chess.Board())
            self.assertEqual(result.evaluation_type, "mate")
            self.assertEqual(result.score, -4)
            self.assertLess(result.target, -0.9)

    def test_v1_and_v2_models_save_and_load_without_dimension_mixing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for input_size in (LEGACY_INPUT_SIZE, INPUT_SIZE):
                path = Path(temporary) / f"model-{input_size}.npz"
                model = MLP(input_size=input_size, seed=9)
                model.save(path)
                loaded = MLP.load(path)
                self.assertEqual(loaded.W1.shape[0], input_size)
                np.testing.assert_array_equal(
                    loaded.forward(np.zeros(input_size)), model.forward(np.zeros(input_size))
                )
        original_model = Path(__file__).with_name("models") / "model_v1_original.npz"
        self.assertEqual(MLP.load(original_model).W1.shape[0], LEGACY_INPUT_SIZE)

    def test_api_inference_uses_the_matching_encoder_for_each_version(self) -> None:
        original_model = api.model
        try:
            board = chess.Board()
            for input_size in (LEGACY_INPUT_SIZE, INPUT_SIZE):
                api.model = MLP(input_size=input_size, seed=10)
                self.assertEqual(api.encode_for_model(board).shape, (input_size,))
                white_score, turn_score = api.get_evaluation(board)
                self.assertTrue(np.isfinite(white_score))
                self.assertEqual(white_score, turn_score)
        finally:
            api.model = original_model

    def test_api_can_evaluate_and_play_with_both_models_in_one_process(self) -> None:
        self.assertIn("v1", api.health()["available_models"])
        self.assertIn("v2", api.health()["available_models"])
        for version in ("v1", "v2"):
            request = api.PositionRequest(
                fen=chess.STARTING_FEN,
                model_version=version,
            )
            evaluation = api.evaluate_position(request)
            self.assertEqual(evaluation["model_version"], version)
            self.assertTrue(np.isfinite(evaluation["white_score"]))
            best_move = api.best_move(request)
            self.assertEqual(best_move["model_version"], version)
            self.assertIn(
                best_move["uci"],
                {move.uci() for move in chess.Board().legal_moves},
            )

        with self.assertRaisesRegex(api.HTTPException, "unavailable"):
            api.evaluate_position(
                api.PositionRequest(fen=chess.STARTING_FEN, model_version="v3")
            )

    def test_checkpoint_round_trip_and_rejects_legacy_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.npz"
            model = MLP(seed=8)
            save_checkpoint(
                model,
                path,
                epoch=2,
                games_processed=40,
                positions_processed=120,
                training_loss_sum=12.5,
                training_count=100,
                completed=False,
                configuration="test",
            )
            loaded, state = load_checkpoint(path)
            self.assertEqual(loaded.W1.shape, (INPUT_SIZE, 256))
            self.assertEqual(state["games_processed"], 40)
            self.assertEqual(state["positions_processed"], 120)
            self.assertEqual(state["training_count"], 100)
            self.assertEqual(state["configuration"], "test")

            legacy_path = Path(temporary) / "legacy.npz"
            MLP(input_size=LEGACY_INPUT_SIZE).save(legacy_path)
            with self.assertRaisesRegex(ValueError, "legacy V1"):
                load_checkpoint(legacy_path)

    def test_synthetic_pgn_training_and_resume_use_separate_held_out_game(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pgn_path = root / "tiny.pgn"
            chunks = []
            for _ in range(5):
                game = chess.pgn.Game()
                node = game
                for uci in ("e2e4", "e7e5", "g1f3"):
                    move = chess.Move.from_uci(uci)
                    node = node.add_variation(move)
                chunks.append(
                    game.accept(
                        chess.pgn.StringExporter(
                            headers=True, variations=False, comments=False
                        )
                    )
                )
            pgn_path.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
            output_path = root / "model_v2.npz"
            checkpoint_path = root / "checkpoint_v2.npz"
            common = {
                "max_games": 5,
                "samples_per_game": 1,
                "epochs": 1,
                "seed": 13,
                "depth": 4,
                "checkpoint_interval": 1,
                "progress_interval": 100,
            }
            with contextlib.redirect_stdout(io.StringIO()):
                train(
                    pgn_path,
                    output_path,
                    checkpoint_path,
                    evaluator=FakeTeacher(),
                    **common,
                )
            model, state = load_checkpoint(checkpoint_path)
            self.assertEqual(model.W1.shape[0], INPUT_SIZE)
            self.assertFalse(state["completed"])
            self.assertEqual(state["training_count"], 4)
            self.assertEqual(state["positions_processed"], 5)

            with contextlib.redirect_stdout(io.StringIO()):
                train(
                    pgn_path,
                    output_path,
                    checkpoint_path,
                    evaluator=FakeTeacher(),
                    max_games=6,
                    samples_per_game=1,
                    epochs=1,
                    seed=13,
                    depth=4,
                    checkpoint_interval=1,
                    progress_interval=100,
                    resume=True,
                )
            _, resumed_state = load_checkpoint(checkpoint_path)
            self.assertTrue(resumed_state["completed"])
            self.assertTrue(output_path.is_file())

    def test_v1_and_v2_regression_use_the_same_synthetic_validation_positions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pgn_path = Path(temporary) / "held-out.pgn"
            one_game = '[Event "test"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 *\n\n'
            pgn_path.write_text(one_game * 5, encoding="utf-8")
            results = evaluate_regression(
                pgn_path,
                MLP(input_size=LEGACY_INPUT_SIZE, seed=2),
                MLP(seed=3),
                FakeTeacher(),
                max_games=5,
                max_positions=1,
                samples_per_game=1,
            )
            self.assertEqual(results["positions_attempted"], 1)
            self.assertEqual(results["v1"]["positions"], 1)
            self.assertEqual(results["v2"]["positions"], 1)
            self.assertEqual(results["skipped_positions"], 0)

    def test_self_play_benchmark_swaps_colors_and_reports_game_results(self) -> None:
        results = play_benchmarks(
            MLP(input_size=LEGACY_INPUT_SIZE, seed=4),
            MLP(seed=5),
            self_play_pairs=1,
            max_plies=1,
        )
        self.assertEqual(results["v1_vs_v2"]["games"], 2)
        self.assertEqual(results["v2_vs_v1"]["games"], 2)
        self.assertEqual(results["v1_vs_v2"]["average_game_length_plies"], 1)
        self.assertEqual(results["v2_vs_v1"]["average_game_length_plies"], 1)


if __name__ == "__main__":
    unittest.main()
