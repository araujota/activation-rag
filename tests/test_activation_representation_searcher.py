import unittest

import numpy as np

from scripts.train_activation_representation_searcher import SearchExample, _metrics_from_scores, evaluate_examples


class ConstantModel:
    def eval(self):
        return None

    def __call__(self, _query):
        import torch

        return torch.tensor([0.0, 1.0], dtype=torch.float32)


class ActivationRepresentationSearcherTests(unittest.TestCase):
    def test_metrics_from_scores_prefers_model_order(self):
        rows = {
            "q1": [
                ("pos", "pos-chunk", 2.0, 2.0, 1),
                ("neg", "neg-chunk", 1.0, 1.0, 0),
            ]
        }

        dense = _metrics_from_scores(rows, top_k=1, use_dense=True)
        model = _metrics_from_scores(rows, top_k=1, use_dense=False)

        self.assertEqual(dense["mrr@1"], 0.0)
        self.assertEqual(model["mrr@1"], 1.0)

    def test_evaluate_examples_scores_predicted_answer_vector(self):
        example = SearchExample(
            query_id="q1",
            query_vector=np.array([1.0, 0.0]),
            candidate_vectors=np.array([[0.0, 1.0], [1.0, 0.0]]),
            labels=np.array([1, 0]),
            doc_ids=("pos", "neg"),
            chunk_ids=("pos-chunk", "neg-chunk"),
            dense_scores=np.array([0.1, 0.9]),
            dense_ranks=np.array([2.0, 1.0]),
        )
        normalizer = {"mean": [0.0, 0.0], "scale": [1.0, 1.0]}

        metrics, scores = evaluate_examples(
            [example],
            model=ConstantModel(),
            normalizer=normalizer,
            top_k=1,
            blend_alpha=None,
            device="cpu",
        )

        self.assertEqual(metrics["model"]["mrr@1"], 1.0)
        self.assertGreater(scores[("q1", "pos-chunk")], scores[("q1", "neg-chunk")])


if __name__ == "__main__":
    unittest.main()
