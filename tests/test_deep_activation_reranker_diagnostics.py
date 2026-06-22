import unittest

from scripts.deep_activation_reranker_diagnostics import pairwise_auc, source_prefix


class DeepActivationRerankerDiagnosticsTest(unittest.TestCase):
    def test_source_prefix_strips_span_offsets(self):
        doc_id = "contractnli:contractnli/example.txt:100:200:0"

        self.assertEqual(source_prefix(doc_id), "contractnli:contractnli/example.txt")

    def test_source_prefix_preserves_plain_doc_ids(self):
        self.assertEqual(source_prefix("123456"), "123456")

    def test_pairwise_auc_counts_ties(self):
        values = [(2.0, True), (1.0, False), (2.0, False)]

        self.assertEqual(pairwise_auc(values), 0.75)


if __name__ == "__main__":
    unittest.main()
