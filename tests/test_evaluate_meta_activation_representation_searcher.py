import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.evaluate_meta_activation_representation_searcher import evaluate_meta_searcher
from scripts.train_meta_activation_representation_searcher import ResidualActivationPredictor


class EvaluateMetaActivationRepresentationSearcherTests(unittest.TestCase):
    def test_evaluates_dense_rerank_and_pure_activation_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = root / "groups.jsonl"
            groups.write_text(json.dumps(_group()) + "\n", encoding="utf-8")
            cache = root / "cache"
            cache.mkdir()
            _write_cache(cache, "query-q1", [1.0, 0.0])
            _write_cache(cache, "pos", [1.0, 0.0])
            _write_cache(cache, "neg", [0.0, 1.0])
            model_path = root / "model.pt"
            _write_model(model_path)

            summary = evaluate_meta_searcher(
                groups_path=groups,
                telemetry_cache_dir=cache,
                model_path=model_path,
                out_path=root / "metrics.json",
                scores_out=root / "scores.jsonl",
                pure_index_scores_out=root / "pure.jsonl",
                top_k=1,
                device="cpu",
            )

            self.assertEqual(summary["query_count"], 1)
            self.assertEqual(summary["metrics"]["pure_dense_candidates"]["mrr@1"], 0.0)
            self.assertEqual(summary["metrics"]["dense_candidates_actpred_rerank"]["mrr@1"], 1.0)
            self.assertEqual(summary["metrics"]["pure_actpred_available_activation_index"]["mrr@1"], 1.0)
            self.assertTrue((root / "scores.jsonl").exists())
            self.assertTrue((root / "pure.jsonl").exists())


def _group() -> dict:
    return {
        "query_id": "q1",
        "query_activation_chunk_id": "query-q1",
        "candidates": [
            {"chunk_id": "neg", "doc_id": "neg-doc", "label": 0, "dense_rank": 1, "dense_score": 1.0},
            {"chunk_id": "pos", "doc_id": "pos-doc", "label": 1, "dense_rank": 2, "dense_score": 0.5},
        ],
    }


def _write_cache(cache: Path, chunk_id: str, values: list[float]) -> None:
    row = {
        "chunk_id": chunk_id,
        "telemetry_valid": True,
        "invalid_reason": None,
        "sae_feature_values": {str(index + 1): value for index, value in enumerate(values)},
    }
    (cache / f"{chunk_id}.json").write_text(json.dumps(row), encoding="utf-8")


def _write_model(path: Path) -> None:
    model = ResidualActivationPredictor(input_dim=2, hidden_dim=2, dropout=0.0)
    state = model.state_dict()
    for key, value in state.items():
        state[key] = torch.zeros_like(value)
    state["residual_scale"] = torch.tensor(1.0)
    payload = {
        "state_dict": state,
        "feature_set_id": "toy",
        "feature_ids": ["1", "2"],
        "representation": "raw",
        "geometry": {
            "geometry_policy": "zscore_center_top_pc_l2",
            "mean": [0.0, 0.0],
            "scale": [1.0, 1.0],
            "components": [],
            "component_variance": [],
            "output_dim": 2,
        },
        "input_dim": 2,
        "hidden_dim": 2,
        "dropout": 0.0,
    }
    torch.save(payload, path)


if __name__ == "__main__":
    unittest.main()
