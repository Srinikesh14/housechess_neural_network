"""Offline tests for safe WSP inspection, candidate edits, and acceptance gates."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import chess.pgn
import numpy as np

from backend.encoder import INPUT_SIZE
from backend.mlp import MLP
from backend.stockfish_eval import PositionEvaluation
from backend.wsp import (
    WSPTools,
    _sha256_file,
    accept_candidate,
    apply_proposal,
    benchmark_playing_strength,
    evaluate_candidate,
    load_dataset,
    OpenRouterClient,
    prepare_datasets,
    run_analysis,
)


def _write_dataset(
    root: Path, split: str, target: float = 0.01, positions: int = 1
) -> Path:
    path = root / f"{split}.jsonl"
    record = {
        "fen": chess.Board().fen(en_passant="fen"),
        "game_id": 25 if split == "sealed" else 5,
        "position_id": 0,
        "target": target,
        "evaluation_type": "centipawn",
        "score": 4.0,
    }
    path.write_text(
        "".join(
            json.dumps({**record, "position_id": index}) + "\n"
            for index in range(positions)
        ),
        encoding="utf-8",
    )
    manifest = {
        "datasets": {
            split: {
                "path": path.name,
                "sha256": _sha256_file(path),
                "positions": positions,
            }
        }
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


class FakeTeacher:
    cache_hits = 0
    new_evaluations = 0
    engine_signature = "synthetic-wsp-teacher"

    def __enter__(self) -> FakeTeacher:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def evaluate(self, _board: chess.Board) -> PositionEvaluation:
        self.new_evaluations += 1
        return PositionEvaluation("centipawn", 4.0, float(np.tanh(4.0 / 400.0)))


class ScriptedLLM:
    prompt_tokens = 20
    completion_tokens = 10
    last_instance: ScriptedLLM | None = None

    def __init__(self, _model: str) -> None:
        self.calls = 0
        self.requests = []
        type(self).last_instance = self

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.requests.append(messages)
        self.calls += 1
        if self.calls == 1:
            return '{"action":"tool","name":"inspect_architecture","arguments":{}}'
        return (
            '{"action":"final","report":{"observations":["synthetic"],'
            '"evidence":[],"hypotheses":[],"uncertainties":[],"next_experiment":""}}'
        )


class WeightSpaceProgrammingTests(unittest.TestCase):
    def test_tools_are_bounded_and_proposal_does_not_change_baseline(self) -> None:
        model = MLP(seed=1)
        path = Path("proposal-not-written.json")
        before = model.W1.copy()
        tools = WSPTools(
            model,
            [{"fen": chess.Board().fen(), "target": 0.0}],
            proposal_path=path,
            allow_proposal=True,
        )
        architecture = tools.dispatch("inspect_architecture", {})
        self.assertEqual(architecture["architecture"][0]["width"], INPUT_SIZE)
        activation = tools.dispatch(
            "analyze_activations",
            {"layer": "hidden1", "neuron_index": 0, "max_positions": 1},
        )
        self.assertEqual(activation["positions"], 1)
        self.assertTrue(np.isfinite(activation["mean"]))
        with self.assertRaises(ValueError):
            tools.dispatch("inspect_neuron", {"layer": "hidden1", "neuron_index": 999})

        proposal = tools.propose_weight_edit(
            parameter="W1",
            coordinates=[[0, 0]],
            operation="add",
            value=0.01,
            reason="synthetic test",
            expected_effect="small local change",
            potential_side_effects="may affect activation",
        )
        self.assertFalse(proposal["applied"])
        np.testing.assert_array_equal(model.W1, before)

    def test_candidate_edit_requires_approval_and_never_overwrites_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev_path = _write_dataset(root, "development")
            model = MLP(seed=3)
            for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
                getattr(model, name).fill(0.0)
            baseline_path = root / "baseline.npz"
            model.save(baseline_path)
            original_hash = _sha256_file(baseline_path)
            proposal_path = root / "proposal.json"
            proposal_path.write_text(
                json.dumps(
                    {
                        "baseline_sha256": original_hash,
                        "parameter": "b3",
                        "coordinates": [[0]],
                        "operation": "set",
                        "value": 0.01,
                        "reason": "correct a measured constant offset",
                        "expected_effect": "reduce this fixed set's error",
                        "potential_side_effects": "may shift all predictions",
                    }
                ),
                encoding="utf-8",
            )
            candidate_path = root / "candidate.npz"
            with self.assertRaisesRegex(ValueError, "approve-candidate-edit"):
                apply_proposal(
                    proposal_path,
                    baseline_path,
                    dev_path,
                    candidate_path,
                    approved=False,
                )
            result = apply_proposal(
                proposal_path,
                baseline_path,
                dev_path,
                candidate_path,
                approved=True,
            )
            self.assertFalse(result["accepted"])
            self.assertLess(
                result["candidate_metrics"]["mse"],
                result["baseline_metrics"]["mse"],
            )
            self.assertEqual(_sha256_file(baseline_path), original_hash)
            self.assertEqual(MLP.load(candidate_path).b3[0], 0.01)

            sealed_path = _write_dataset(root, "sealed", positions=100)
            report_path = root / "sealed-report.json"
            evaluate_candidate(
                baseline_path,
                candidate_path,
                sealed_path,
                report_path,
                "sealed",
            )
            with self.assertRaisesRegex(ValueError, "already been used"):
                evaluate_candidate(
                    baseline_path,
                    candidate_path,
                    sealed_path,
                    root / "second-sealed-report.json",
                    "sealed",
                )
            accepted_path = root / "accepted.npz"
            accepted = accept_candidate(
                report_path,
                accepted_path,
                confirmed=True,
            )
            self.assertTrue(accepted_path.is_file())
            self.assertIn("accepted_model", accepted)

    def test_edit_limits_reject_large_or_out_of_range_changes(self) -> None:
        tools = WSPTools(
            MLP(seed=4),
            [{"fen": chess.Board().fen(), "target": 0.0}],
            proposal_path=Path("unused.json"),
            allow_proposal=True,
        )
        common = {
            "parameter": "b1",
            "coordinates": [[0]],
            "operation": "add",
            "reason": "reason",
            "expected_effect": "effect",
            "potential_side_effects": "risks",
        }
        with self.assertRaisesRegex(ValueError, "at most"):
            tools.propose_weight_edit(value=0.1, **common)
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            tools.propose_weight_edit(
                value=0.001,
                **{**common, "coordinates": [[999]]},
            )

    def test_acceptance_rejects_non_sealed_report_and_baseline_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = MLP(seed=5)
            baseline_path = root / "baseline.npz"
            baseline.save(baseline_path)
            candidate_path = root / "candidate.npz"
            baseline.save(candidate_path)
            dev_path = _write_dataset(root, "development", target=0.0)
            report_path = root / "development-report.json"
            evaluate_candidate(
                baseline_path, candidate_path, dev_path, report_path, "development"
            )
            with self.assertRaisesRegex(ValueError, "sealed-set"):
                accept_candidate(
                    report_path,
                    root / "accepted.npz",
                    confirmed=True,
                )
            with self.assertRaisesRegex(ValueError, "explicit"):
                accept_candidate(
                    report_path,
                    root / "accepted.npz",
                    confirmed=False,
                )

    def test_dataset_builder_keeps_development_and_sealed_games_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pgn_path = root / "games.pgn"
            games = []
            for _ in range(25):
                game = chess.pgn.Game()
                game.add_variation(chess.Move.from_uci("e2e4"))
                games.append(
                    game.accept(
                        chess.pgn.StringExporter(
                            headers=True, variations=False, comments=False
                        )
                    )
                )
            pgn_path.write_text("\n\n".join(games) + "\n", encoding="utf-8")
            output_dir = root / "datasets"
            with patch("backend.wsp.StockfishEvaluator", return_value=FakeTeacher()):
                manifest = prepare_datasets(
                    pgn_path,
                    output_dir,
                    stockfish="fake",
                    cache_path=root / "cache.sqlite3",
                    max_games=25,
                    development_after_game=0,
                    sealed_after_game=20,
                    max_positions_per_split=1,
                    samples_per_game=1,
                )
            development = load_dataset(output_dir / "development.jsonl")
            sealed = load_dataset(output_dir / "sealed.jsonl")
            self.assertEqual(development[0]["game_id"], 5)
            self.assertEqual(sealed[0]["game_id"], 25)
            self.assertNotEqual(development[0]["game_id"], sealed[0]["game_id"])
            self.assertEqual(manifest["counts"], {"development": 1, "sealed": 1})

    def test_candidate_playing_benchmark_pairs_colors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_path = root / "baseline.npz"
            candidate_path = root / "candidate.npz"
            MLP(seed=8).save(baseline_path)
            MLP(seed=9).save(candidate_path)
            report = benchmark_playing_strength(
                baseline_path,
                candidate_path,
                self_play_pairs=1,
                max_plies=1,
            )
            self.assertEqual(
                report["results"]["baseline_vs_candidate"]["games"], 2
            )
            self.assertEqual(
                report["results"]["candidate_vs_baseline"]["games"], 2
            )

    def test_openrouter_client_uses_configured_model_and_reports_usage(self) -> None:
        payload = {
            "choices": [{"message": {"content": '{"action":"final","report":{}}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "backend.wsp.urllib.request.urlopen",
                return_value=io.BytesIO(json.dumps(payload).encode("utf-8")),
            ) as urlopen:
                client = OpenRouterClient("test/provider-model")
                response = client.complete([{"role": "user", "content": "test"}])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertIn(b"test/provider-model", request.data)
        self.assertEqual(response, '{"action":"final","report":{}}')
        self.assertEqual(client.prompt_tokens, 12)
        self.assertEqual(client.completion_tokens, 4)

    def test_read_only_llm_workflow_uses_local_tools_and_preserves_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = _write_dataset(root, "development")
            model_path = root / "baseline.npz"
            model = MLP(seed=22)
            model.save(model_path)
            before_hash = _sha256_file(model_path)
            client = ScriptedLLM
            with patch("backend.wsp.OpenRouterClient", client):
                run_dir = run_analysis(
                    model_path,
                    dataset_path,
                    root / "experiments",
                    model_id="test/mock",
                    max_steps=2,
                )
            result = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
            self.assertEqual(result["mode"], "read-only")
            self.assertEqual(result["termination_reason"], "final_report_received")
            self.assertIsNone(result["proposal_path"])
            self.assertIsNotNone(client.last_instance)
            self.assertNotIn(
                '"name": "propose_weight_edit"',
                client.last_instance.requests[0][0]["content"],
            )
            self.assertEqual(_sha256_file(model_path), before_hash)


if __name__ == "__main__":
    unittest.main()
