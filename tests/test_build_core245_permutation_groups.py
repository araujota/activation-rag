import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_core245_permutation_groups import build_permutation_groups


class BuildCore245PermutationGroupsTests(unittest.TestCase):
    def test_builds_raw_log_topk_and_df_filtered_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups_path, cache_dir, manifest = _fixture(root)

            raw = build_permutation_groups(
                groups_path=groups_path,
                telemetry_cache_dir=cache_dir,
                feature_manifest_path=manifest,
                variant="qwen_l07_core245_raw_max",
                out_path=root / "raw.jsonl",
                top_effect_k=2,
                df_min_fraction=0.0,
                df_max_fraction=1.0,
                counterfactual_seed=13,
            )
            log = build_permutation_groups(
                groups_path=groups_path,
                telemetry_cache_dir=cache_dir,
                feature_manifest_path=manifest,
                variant="qwen_l07_core245_log1p_l2",
                out_path=root / "log.jsonl",
                top_effect_k=2,
                df_min_fraction=0.0,
                df_max_fraction=1.0,
                counterfactual_seed=13,
            )
            topk = build_permutation_groups(
                groups_path=groups_path,
                telemetry_cache_dir=cache_dir,
                feature_manifest_path=manifest,
                variant="qwen_l07_core245_high_effect_topk",
                out_path=root / "topk.jsonl",
                top_effect_k=2,
                df_min_fraction=0.0,
                df_max_fraction=1.0,
                counterfactual_seed=13,
            )
            df_filtered = build_permutation_groups(
                groups_path=groups_path,
                telemetry_cache_dir=cache_dir,
                feature_manifest_path=manifest,
                variant="qwen_l07_core245_df_filtered",
                out_path=root / "df.jsonl",
                top_effect_k=2,
                df_min_fraction=0.5,
                df_max_fraction=0.99,
                counterfactual_seed=13,
            )

        raw_features = raw["groups"][0]["candidates"][0]["features"]
        log_features = log["groups"][0]["candidates"][0]["features"]
        topk_features = topk["groups"][0]["candidates"][0]["features"]
        df_features = df_filtered["groups"][0]["candidates"][0]["features"]
        self.assertIn("core245:1:product", raw_features)
        self.assertIn("core245_matcher:csls", raw_features)
        self.assertIn("core245_matcher:nicdm", raw_features)
        self.assertIn("core245_log1p_l2:1:product", log_features)
        self.assertIn("core245_high_effect:3:product", topk_features)
        self.assertNotIn("core245_high_effect:2:product", topk_features)
        self.assertIn("core245_df:1:product", df_features)
        self.assertNotIn("core245_df:2:product", df_features)

    def test_builds_category_and_matched_counterfactual_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups_path, cache_dir, manifest = _fixture(root)

            category = build_permutation_groups(
                groups_path=groups_path,
                telemetry_cache_dir=cache_dir,
                feature_manifest_path=manifest,
                variant="qwen_l07_core245_causal_weighted_category",
                out_path=root / "category.jsonl",
                top_effect_k=2,
                df_min_fraction=0.0,
                df_max_fraction=1.0,
                counterfactual_seed=13,
            )
            counterfactual = build_permutation_groups(
                groups_path=groups_path,
                telemetry_cache_dir=cache_dir,
                feature_manifest_path=manifest,
                variant="qwen_l07_core245_counterfactual_matched",
                out_path=root / "counterfactual.jsonl",
                top_effect_k=2,
                df_min_fraction=0.0,
                df_max_fraction=1.0,
                counterfactual_seed=13,
            )

        category_features = category["groups"][0]["candidates"][0]["features"]
        counterfactual_features = counterfactual["groups"][0]["candidates"][0]["features"]
        self.assertIn("core245_category:evidence:weighted_product", category_features)
        self.assertIn("core245_category:format:cosine", category_features)
        self.assertIn("core245_counterfactual:evidence:weighted_product", counterfactual_features)
        self.assertEqual(category["summary"]["variant"], "qwen_l07_core245_causal_weighted_category")
        self.assertEqual(counterfactual["summary"]["counterfactual_seed"], 13)


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    cache_dir = root / "cache"
    cache_dir.mkdir()
    manifest = root / "feature_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "vicuna.rmt_span_selector.feature_manifest.v1",
                "feature_set_id": "fixture-core245",
                "features": [
                    {"feature_id": "1", "label": "answer evidence", "categories": ["evidence"], "causal_effect": 5.0},
                    {"feature_id": "2", "label": "ubiquitous prior", "categories": ["format"], "causal_effect": 1.0},
                    {"feature_id": "3", "label": "task feature", "categories": ["format"], "causal_effect": 4.0},
                ],
                "sae_feature_ids": ["1", "2", "3"],
            }
        ),
        encoding="utf-8",
    )
    _write_cache(cache_dir, "query-chunk", "query", {"1": 2.0, "2": 1.0, "3": 0.0})
    _write_cache(cache_dir, "doc-pos", "doc-pos", {"1": 2.0, "2": 1.0, "3": 0.5})
    _write_cache(cache_dir, "doc-neg", "doc-neg", {"1": 0.0, "2": 1.0, "3": 2.0})
    groups_path = root / "groups.jsonl"
    groups_path.write_text(
        json.dumps(
            {
                "schema_version": "activation_rag.supervised_reranker_group.v1",
                "dataset_name": "fixture",
                "split": "test",
                "query_id": "q1",
                "query_text": "evidence",
                "query_activation_chunk_id": "query-chunk",
                "candidate_k": 2,
                "positive_doc_ids": ["doc-pos"],
                "positive_in_candidate_pool": True,
                "false_negative_policy": "fixture",
                "candidates": [
                    {"chunk_id": "doc-pos", "doc_id": "doc-pos", "label": 1, "dense_rank": 1, "dense_score": 0.9, "features": {"dense_score": 0.9, "dense_rank_reciprocal": 1.0}},
                    {"chunk_id": "doc-neg", "doc_id": "doc-neg", "label": 0, "dense_rank": 2, "dense_score": 0.8, "features": {"dense_score": 0.8, "dense_rank_reciprocal": 0.5}},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return groups_path, cache_dir, manifest


def _write_cache(cache_dir: Path, chunk_id: str, document_id: str, values: dict[str, float]) -> None:
    (cache_dir / f"{chunk_id}.json").write_text(
        json.dumps(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "telemetry_valid": True,
                "sae_feature_values": values,
            }
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
