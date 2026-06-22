import unittest

from scripts.run_semantic_activation_feature_sweep import build_feature_variants


class SemanticActivationSweepScriptTests(unittest.TestCase):
    def test_build_feature_variants_groups_semantic_and_counterfactual_features(self):
        groups = [
            {
                "query_id": "q1",
                "candidates": [
                    {
                        "features": {
                            "dense_score": 0.9,
                            "dense_rank_reciprocal": 1.0,
                            "activation_semantic:semantic_label:answer_evidence:cosine": 1.0,
                            "activation_semantic:category:evidence:abs_diff_mean": 0.1,
                            "activation_semantic:causal_confidence:high:product_mean": 0.5,
                            "activation_semantic:counterfactual:shuffle_00:cosine": -0.2,
                            "activation_cosine": 0.3,
                        }
                    }
                ],
            }
        ]

        variants = build_feature_variants(groups)

        self.assertIn("dense_only", variants)
        self.assertEqual(variants["dense_only"], ("dense_rank_reciprocal", "dense_score"))
        self.assertIn("dense_plus_semantic_labels", variants)
        self.assertIn("activation_semantic:semantic_label:answer_evidence:cosine", variants["dense_plus_semantic_labels"])
        self.assertIn("dense_plus_categories", variants)
        self.assertIn("activation_semantic:category:evidence:abs_diff_mean", variants["dense_plus_categories"])
        self.assertIn("dense_plus_causal_confidence", variants)
        self.assertIn("activation_semantic:causal_confidence:high:product_mean", variants["dense_plus_causal_confidence"])
        self.assertIn("dense_plus_counterfactual", variants)
        self.assertIn("activation_semantic:counterfactual:shuffle_00:cosine", variants["dense_plus_counterfactual"])


if __name__ == "__main__":
    unittest.main()
