from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from activation_rag.supervised_reranking import load_jsonl
from scripts.merge_reranker_groups import merge_group_files


class MergeRerankerGroupsTests(unittest.TestCase):
    def test_merge_stamps_combined_dataset_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.jsonl"
            b = root / "b.jsonl"
            out = root / "out.jsonl"
            _write_jsonl(a, [{"query_id": "q1", "dataset_name": "source-a", "positive_in_candidate_pool": True}])
            _write_jsonl(b, [{"query_id": "q2", "dataset_name": "source-b", "positive_in_candidate_pool": False}])

            summary = merge_group_files(input_paths=[a, b], out_path=out, dataset_name="pooled")
            rows = load_jsonl(out)

        self.assertEqual(2, summary["group_count"])
        self.assertEqual(["pooled", "pooled"], [row["dataset_name"] for row in rows])
        self.assertEqual(["source-a", "source-b"], [row["source_dataset_name"] for row in rows])

    def test_merge_rejects_duplicate_query_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.jsonl"
            b = root / "b.jsonl"
            _write_jsonl(a, [{"query_id": "q1"}])
            _write_jsonl(b, [{"query_id": "q1"}])

            with self.assertRaises(ValueError):
                merge_group_files(input_paths=[a, b], out_path=root / "out.jsonl")

    def test_merge_can_prefix_colliding_query_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.jsonl"
            b = root / "b.jsonl"
            out = root / "out.jsonl"
            _write_jsonl(a, [{"query_id": "1", "dataset_name": "source-a"}])
            _write_jsonl(b, [{"query_id": "1", "dataset_name": "source-b"}])

            merge_group_files(input_paths=[a, b], out_path=out, dataset_name="pooled", prefix_query_ids=True)
            rows = load_jsonl(out)

        self.assertEqual(["source-a:1", "source-b:1"], [row["query_id"] for row in rows])
        self.assertEqual(["1", "1"], [row["source_query_id"] for row in rows])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
