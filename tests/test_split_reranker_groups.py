import json
import tempfile
import unittest
from pathlib import Path

from scripts.split_reranker_groups import split_groups


class SplitRerankerGroupsTests(unittest.TestCase):
    def test_split_groups_is_deterministic_and_disjoint(self):
        groups = [{"query_id": f"q{i}", "candidates": []} for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "groups.jsonl"
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in groups) + "\n", encoding="utf-8")

            summary = split_groups(source=source, train_out=train, dev_out=dev, dev_fraction=0.25, seed="fixture")

            train_rows = _read(train)
            dev_rows = _read(dev)

        self.assertEqual(summary["input_count"], 20)
        self.assertEqual(len(dev_rows), 5)
        self.assertEqual(len(train_rows), 15)
        self.assertFalse({row["query_id"] for row in train_rows} & {row["query_id"] for row in dev_rows})


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
