import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_reranker_comparison import run_audit


class AuditRerankerComparisonScriptTests(unittest.TestCase):
    def test_run_audit_reads_score_jsonl_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = root / "groups.jsonl"
            scores = root / "scores.jsonl"
            out = root / "audit.json"
            _write_jsonl(groups, [_group()])
            _write_jsonl(
                scores,
                [
                    {"query_id": "q1", "chunk_id": "neg", "score": 0.1},
                    {"query_id": "q1", "chunk_id": "pos", "score": 0.9},
                ],
            )

            summary = run_audit(
                groups_path=groups,
                out_path=out,
                baseline_name="dense",
                candidate_name="reranker",
                baseline_scores_path=None,
                candidate_scores_path=scores,
                candidate_mlp_checkpoint=None,
                candidate_scores_out=None,
                top_k=1,
                randomization_iterations=100,
                changed_query_limit=1,
                seed=5,
            )

            self.assertTrue(out.exists())
            self.assertEqual(summary["candidate_name"], "reranker")
            self.assertEqual(summary["candidate_metrics"]["ndcg@1"], 1.0)
            self.assertEqual(summary["baseline_metrics"]["ndcg@1"], 0.0)


def _group() -> dict:
    return {
        "query_id": "q1",
        "query_text": "test query",
        "candidates": [
            {"chunk_id": "neg", "doc_id": "neg-doc", "label": 0, "dense_rank": 1, "text": "negative"},
            {"chunk_id": "pos", "doc_id": "pos-doc", "label": 1, "dense_rank": 2, "text": "positive"},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
