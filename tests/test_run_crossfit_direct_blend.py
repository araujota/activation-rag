from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_crossfit_direct_blend import concat_score_files, split_rows_into_folds


class RunCrossfitDirectBlendTests(unittest.TestCase):
    def test_split_rows_into_folds_is_query_disjoint_and_deterministic(self) -> None:
        rows = [{"query_id": f"q{i}"} for i in range(17)]

        first = split_rows_into_folds(rows, folds=5, seed="seed")
        second = split_rows_into_folds(rows, folds=5, seed="seed")

        self.assertEqual([[row["query_id"] for row in fold] for fold in first], [[row["query_id"] for row in fold] for fold in second])
        flattened = [row["query_id"] for fold in first for row in fold]
        self.assertEqual(set(row["query_id"] for row in rows), set(flattened))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertLessEqual(max(len(fold) for fold in first) - min(len(fold) for fold in first), 1)

    def test_concat_score_files_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.jsonl"
            b = root / "b.jsonl"
            out = root / "out.jsonl"
            a.write_text(json.dumps({"query_id": "q1", "chunk_id": "c1", "score": 1.0}) + "\n", encoding="utf-8")
            b.write_text(json.dumps({"query_id": "q1", "chunk_id": "c1", "score": 2.0}) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                concat_score_files([a, b], out)


if __name__ == "__main__":
    unittest.main()
