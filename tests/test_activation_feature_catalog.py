import json
import tempfile
import unittest
from pathlib import Path

from activation_rag.feature_catalog import load_feature_catalog


class ActivationFeatureCatalogTests(unittest.TestCase):
    def test_loads_pattern_catalog_and_groups_matching_activation_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "activation_rag.activation_feature_catalog.v1",
                        "features": [
                            {
                                "feature_pattern": "act:site_a:*:mean",
                                "semantic_label": "answer_evidence",
                                "categories": ["evidence"],
                                "polarity": "positive",
                                "causal_confidence": "high",
                            },
                            {
                                "feature_pattern": "act:site_b:*:*",
                                "semantic_label": "formatting_pressure",
                                "categories": ["format"],
                                "polarity": "mixed",
                                "causal_confidence": "medium",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            catalog = load_feature_catalog(path)
            groups = catalog.groups_for_feature_names(
                [
                    "act:site_a:prefill_last:mean",
                    "act:site_a:prefill_last:rms",
                    "act:site_b:post50_mean:max_abs",
                ]
            )

        self.assertEqual(groups["semantic_label:answer_evidence"], ["act:site_a:prefill_last:mean"])
        self.assertEqual(groups["category:evidence"], ["act:site_a:prefill_last:mean"])
        self.assertEqual(groups["causal_confidence:high"], ["act:site_a:prefill_last:mean"])
        self.assertEqual(groups["polarity:positive"], ["act:site_a:prefill_last:mean"])
        self.assertEqual(groups["semantic_label:formatting_pressure"], ["act:site_b:post50_mean:max_abs"])

    def test_loads_longmem_feature_manifest_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feature_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "vicuna.rmt_span_selector.feature_manifest.v1",
                        "feature_set_id": "causal-core",
                        "features": [
                            {
                                "feature_id": "17",
                                "label": "source citation answer evidence",
                                "categories": ["relation_discourse", "task_instruction"],
                                "validation_status": "causal_support",
                                "causal_effect": 0.4,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            catalog = load_feature_catalog(path)
            groups = catalog.groups_for_feature_names(["sae.feature.17", "sae.feature.18"])

        self.assertEqual(groups["semantic_label:source_citation_answer_evidence"], ["sae.feature.17"])
        self.assertEqual(groups["category:relation_discourse"], ["sae.feature.17"])
        self.assertEqual(groups["validation_status:causal_support"], ["sae.feature.17"])

    def test_longmem_feature_manifest_matches_raw_selector_feature_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feature_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "vicuna.rmt_span_selector.feature_manifest.v1",
                        "feature_set_id": "causal-core",
                        "features": [
                            {
                                "feature_id": "5439",
                                "label": "task instruction event action",
                                "categories": ["task_instruction", "event_action"],
                                "validation_status": "causal_support",
                                "causal_effect": 3.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            catalog = load_feature_catalog(path)
            groups = catalog.groups_for_feature_names(["5439", "18172"])

        self.assertEqual(groups["semantic_label:task_instruction_event_action"], ["5439"])
        self.assertEqual(groups["category:task_instruction"], ["5439"])

    def test_can_build_deterministic_counterfactual_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "activation_rag.activation_feature_catalog.v1",
                        "counterfactual_group_count": 2,
                        "counterfactual_group_size": 2,
                        "counterfactual_seed": 7,
                        "features": [
                            {"feature_pattern": "act:site_a:*", "semantic_label": "answer_evidence"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            catalog = load_feature_catalog(path)
            first = catalog.groups_for_feature_names(["act:site_a:x", "act:site_b:x", "act:site_c:x"])
            second = catalog.groups_for_feature_names(["act:site_a:x", "act:site_b:x", "act:site_c:x"])

        self.assertEqual(first["counterfactual:shuffle_00"], second["counterfactual:shuffle_00"])
        self.assertEqual(len(first["counterfactual:shuffle_00"]), 2)
        self.assertEqual(len(first["counterfactual:shuffle_01"]), 2)


if __name__ == "__main__":
    unittest.main()
