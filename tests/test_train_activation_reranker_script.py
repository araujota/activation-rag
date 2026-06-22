import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_activation_reranker import run_training


class TrainActivationRerankerScriptTests(unittest.TestCase):
    def test_run_training_writes_model_and_metrics(self):
        groups = [
            {
                "query_id": "q1",
                "candidates": [
                    {"chunk_id": "n1", "doc_id": "d2", "label": 0, "dense_rank": 1, "features": {"f": 0.0}},
                    {"chunk_id": "p1", "doc_id": "d1", "label": 1, "dense_rank": 2, "features": {"f": 1.0}},
                ],
            },
            {
                "query_id": "q2",
                "candidates": [
                    {"chunk_id": "n2", "doc_id": "d4", "label": 0, "dense_rank": 1, "features": {"f": 0.0}},
                    {"chunk_id": "p2", "doc_id": "d3", "label": 1, "dense_rank": 2, "features": {"f": 1.0}},
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "train.jsonl"
            dev_path = root / "dev.jsonl"
            model_path = root / "model.json"
            metrics_path = root / "metrics.json"
            _write_jsonl(train_path, groups)
            _write_jsonl(dev_path, groups)

            summary = run_training(
                train_path=train_path,
                dev_path=dev_path,
                test_path=None,
                model_out=model_path,
                metrics_out=metrics_path,
                epochs=100,
                learning_rate=0.1,
                l2=0.0,
                top_k=2,
            )

            self.assertGreater(summary["dev_metrics"]["model"]["mrr@2"], summary["dev_metrics"]["dense"]["mrr@2"])
            self.assertTrue(model_path.exists())
            self.assertTrue(metrics_path.exists())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
