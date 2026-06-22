from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from activation_rag.embedding import HashEmbeddingProvider
from scripts.prepare_legalbenchrag_groups import load_legalbenchrag_components
from scripts.prepare_vertical_reranker_groups import prepare_retrieval_groups, split_groups


class PrepareLegalBenchRagGroupsTests(unittest.TestCase):
    def test_loads_span_windows_and_prepares_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_dir = root / "corpus" / "cuad"
            benchmarks_dir = root / "benchmarks"
            corpus_dir.mkdir(parents=True)
            benchmarks_dir.mkdir()
            document_text = "Intro text. " + ("padding " * 80) + "The contract expires on December 31, 2028. " + ("tail " * 80)
            answer_start = document_text.index("The contract expires")
            answer_end = answer_start + len("The contract expires on December 31, 2028.")
            (corpus_dir / "contract.txt").write_text(document_text, encoding="utf-8")
            (benchmarks_dir / "cuad.json").write_text(
                json.dumps(
                    {
                        "tests": [
                            {
                                "query": "When does the contract expire?",
                                "snippets": [
                                    {
                                        "file_path": "cuad/contract.txt",
                                        "span": [answer_start, answer_end],
                                        "answer": "December 31, 2028",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            components = load_legalbenchrag_components(root=root, window_chars=220)
            groups = prepare_retrieval_groups(
                components,
                candidate_k=1,
                embedder=HashEmbeddingProvider(dimension=64),
                append_qrel_positives=True,
            )

        self.assertEqual(1, len(components.queries))
        self.assertEqual(1, len(components.corpus))
        passage_text = next(iter(components.corpus.values()))
        self.assertIn("December 31, 2028", passage_text)
        self.assertLessEqual(len(passage_text), 220)
        self.assertTrue(groups[0]["positive_in_candidate_pool"])
        self.assertEqual(1, groups[0]["candidates"][0]["label"])

    def test_split_is_query_disjoint(self) -> None:
        groups = [
            {"query_id": f"cuad:{index}", "split": "unsplit", "candidates": [{"label": 1}], "positive_in_candidate_pool": True}
            for index in range(30)
        ]

        train_rows, dev_rows, test_rows = split_groups(groups, dev_fraction=0.2, test_fraction=0.2, seed="legal")

        self.assertFalse({row["query_id"] for row in train_rows} & {row["query_id"] for row in dev_rows})
        self.assertFalse({row["query_id"] for row in train_rows} & {row["query_id"] for row in test_rows})
        self.assertFalse({row["query_id"] for row in dev_rows} & {row["query_id"] for row in test_rows})


if __name__ == "__main__":
    unittest.main()
