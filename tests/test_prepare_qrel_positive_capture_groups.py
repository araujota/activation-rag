import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_qrel_positive_capture_groups import prepare_qrel_positive_capture_groups


class PrepareQrelPositiveCaptureGroupsTests(unittest.TestCase):
    def test_streams_only_qrel_positive_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            beir = root / "beir"
            (beir / "qrels").mkdir(parents=True)
            _write_jsonl(
                beir / "corpus.jsonl",
                [
                    {"_id": "unused", "title": "unused", "text": "unused"},
                    {"_id": "d1", "title": "Doc One", "text": "alpha evidence"},
                    {"_id": "d2", "title": "Doc Two", "text": "beta evidence"},
                ],
            )
            _write_jsonl(beir / "queries.jsonl", [{"_id": "q1", "text": "alpha?"}])
            (beir / "qrels" / "train.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\n", encoding="utf-8")

            groups = prepare_qrel_positive_capture_groups(beir_dir=beir, dataset_name="fixture", split="train")

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["schema_version"], "activation_rag.qrel_positive_capture_group.v1")
        self.assertEqual(groups[0]["positive_doc_ids"], ["d1"])
        self.assertEqual(len(groups[0]["candidates"]), 1)
        self.assertEqual(groups[0]["candidates"][0]["doc_id"], "d1")
        self.assertEqual(groups[0]["candidates"][0]["label"], 1)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
