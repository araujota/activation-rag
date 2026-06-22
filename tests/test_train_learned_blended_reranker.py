import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_learned_blended_reranker import run_training


class LearnedBlendedRerankerTests(unittest.TestCase):
    def test_run_training_learns_activation_weight_over_dense(self):
        groups = [
            {
                "query_id": "q1",
                "candidates": [
                    {"chunk_id": "n1", "doc_id": "d2", "label": 0, "dense_rank": 1, "dense_score": 2.0},
                    {"chunk_id": "p1", "doc_id": "d1", "label": 1, "dense_rank": 2, "dense_score": 1.0},
                ],
            },
            {
                "query_id": "q2",
                "candidates": [
                    {"chunk_id": "n2", "doc_id": "d4", "label": 0, "dense_rank": 1, "dense_score": 2.0},
                    {"chunk_id": "p2", "doc_id": "d3", "label": 1, "dense_rank": 2, "dense_score": 1.0},
                ],
            },
        ]
        scores = [
            {"query_id": "q1", "chunk_id": "n1", "score": 0.1},
            {"query_id": "q1", "chunk_id": "p1", "score": 0.9},
            {"query_id": "q2", "chunk_id": "n2", "score": 0.1},
            {"query_id": "q2", "chunk_id": "p2", "score": 0.9},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups_path = root / "groups.jsonl"
            scores_path = root / "scores.jsonl"
            _write_jsonl(groups_path, groups)
            _write_jsonl(scores_path, scores)

            summary = run_training(
                train_groups_path=groups_path,
                dev_groups_path=groups_path,
                test_groups_path=groups_path,
                train_scores_path=scores_path,
                dev_scores_path=scores_path,
                test_scores_path=scores_path,
                model_out=root / "model.json",
                metrics_out=root / "metrics.json",
                test_scores_out=root / "test-scores.jsonl",
                dataset_name="fixture",
                epochs=50,
                learning_rate=0.1,
                l2=0.0,
                max_pairs_per_query=16,
                top_k=2,
                seed=13,
            )

            self.assertGreater(summary["test_metrics"]["model"]["mrr@2"], summary["test_metrics"]["dense"]["mrr@2"])
            self.assertGreater(summary["weights"]["z_actpred"], 0.0)
            self.assertTrue((root / "model.json").exists())
            self.assertTrue((root / "test-scores.jsonl").exists())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
