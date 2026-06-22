import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_activation_mlp_reranker import run_mlp_training


class TrainActivationMlpRerankerTests(unittest.TestCase):
    def test_mlp_training_improves_toy_dev_ranking(self):
        groups = [
            _group("q1", "p1", "n1"),
            _group("q2", "p2", "n2"),
            _group("q3", "p3", "n3"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            model = root / "model.pt"
            metrics = root / "metrics.json"
            _write_jsonl(train, groups)
            _write_jsonl(dev, groups)

            summary = run_mlp_training(
                train_path=train,
                dev_path=dev,
                test_path=None,
                model_out=model,
                metrics_out=metrics,
                feature_names=("activation_match", "dense_score"),
                hidden_dim=8,
                epochs=80,
                learning_rate=0.01,
                weight_decay=0.0,
                loss_name="listwise_softmax",
                top_k=2,
                device="cpu",
                seed=13,
            )

            self.assertGreater(summary["dev_metrics"]["model"]["mrr@2"], summary["dev_metrics"]["dense"]["mrr@2"])
            self.assertEqual(summary["selection_metric"], "ndcg@2")
            self.assertIn("best_dev_score", summary)
            self.assertIn("best_epoch", summary)
            self.assertTrue(model.exists())
            self.assertTrue(metrics.exists())


def _group(query_id: str, positive_chunk: str, negative_chunk: str) -> dict:
    return {
        "query_id": query_id,
        "candidates": [
            {
                "chunk_id": negative_chunk,
                "doc_id": f"{negative_chunk}-doc",
                "label": 0,
                "dense_rank": 1,
                "features": {"activation_match": -1.0, "dense_score": 0.9},
            },
            {
                "chunk_id": positive_chunk,
                "doc_id": f"{positive_chunk}-doc",
                "label": 1,
                "dense_rank": 2,
                "features": {"activation_match": 1.0, "dense_score": 0.7},
            },
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
