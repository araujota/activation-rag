import tempfile
import unittest
import json
from pathlib import Path

from scripts.audit_group_leakage import audit_group_leakage


class AuditGroupLeakageTests(unittest.TestCase):
    def test_audit_reports_exact_query_and_positive_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            test = root / "test.jsonl"
            out = root / "leakage.json"
            write_jsonl(
                train,
                [
                    _group("q1", "What is Aspirin", ["doc-a"], [("doc-a", 1), ("doc-b", 0)]),
                    _group("q2", "unique train", ["doc-c"], [("doc-c", 1)]),
                ],
            )
            write_jsonl(
                test,
                [
                    _group("q3", " what IS aspirin ", ["doc-a"], [("doc-a", 1), ("doc-d", 0)]),
                    _group("q4", "unique test", ["doc-e"], [("doc-e", 1)]),
                ],
            )

            summary = audit_group_leakage(
                train_groups_path=train,
                dev_groups_path=None,
                test_groups_path=test,
                out_path=out,
                example_limit=5,
            )

            comparison = summary["comparisons"]["train_vs_test"]
            self.assertEqual(comparison["query_id_overlap_count"], 0)
            self.assertEqual(comparison["query_text_overlap_count"], 1)
            self.assertEqual(comparison["positive_doc_overlap_count"], 1)
            self.assertEqual(comparison["positive_pair_overlap_count"], 0)
            self.assertEqual(comparison["candidate_text_overlap_count"], 1)
            self.assertEqual(comparison["positive_text_overlap_count"], 1)
            self.assertTrue(out.exists())

    def test_audit_reports_near_duplicate_query_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            test = root / "test.jsonl"
            out = root / "leakage.json"
            write_jsonl(
                train,
                [
                    _group(
                        "q1",
                        "Which drug reduces fever in pediatric patients with influenza symptoms",
                        ["doc-a"],
                        [("doc-a", 1), ("doc-b", 0)],
                    ),
                ],
            )
            write_jsonl(
                test,
                [
                    _group(
                        "q2",
                        "Which drug reduces fever in pediatric patients with influenza infection",
                        ["doc-c"],
                        [("doc-c", 1), ("doc-d", 0)],
                    ),
                ],
            )

            summary = audit_group_leakage(
                train_groups_path=train,
                dev_groups_path=None,
                test_groups_path=test,
                out_path=out,
                example_limit=5,
                near_duplicate_threshold=0.5,
                shingle_size=3,
            )

            comparison = summary["comparisons"]["train_vs_test"]
            self.assertEqual(comparison["query_text_overlap_count"], 0)
            self.assertEqual(comparison["near_duplicate_query_text_count"], 1)


def _group(query_id: str, query_text: str, positive_doc_ids: list[str], docs: list[tuple[str, int]]) -> dict:
    return {
        "query_id": query_id,
        "query_text": query_text,
        "positive_doc_ids": positive_doc_ids,
        "candidates": [
            {
                "chunk_id": f"{query_id}-{doc_id}",
                "doc_id": doc_id,
                "dense_rank": rank,
                "dense_score": 1.0 / rank,
                "label": label,
                "text": doc_id,
            }
            for rank, (doc_id, label) in enumerate(docs, start=1)
        ],
    }

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
