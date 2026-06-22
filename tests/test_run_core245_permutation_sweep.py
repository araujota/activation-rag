import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_core245_permutation_sweep import run_sweep


class RunCore245PermutationSweepTests(unittest.TestCase):
    def test_plan_only_builds_all_ordered_variant_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups, cache, manifest = _fixture(root)

            summary = run_sweep(
                train_groups=groups,
                dev_groups=groups,
                test_groups=groups,
                telemetry_cache_dir=cache,
                feature_manifest=manifest,
                out_dir=root / "sweep",
                manifest_out=root / "manifest.json",
                top_effect_k=1,
                df_min_fraction=0.0,
                df_max_fraction=0.99,
                counterfactual_seed=13,
                hidden_dim=8,
                epochs=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                loss_name="listwise_softmax",
                top_k=2,
                device="cpu",
                seed=13,
                plan_only=True,
            )

            self.assertTrue(summary["plan_only"])
            self.assertEqual(len(summary["variants"]), 6)
            for variant in summary["variants"]:
                self.assertTrue(Path(variant["paths"]["train"]).exists())
                self.assertGreater(variant["feature_count"], 0)


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    cache = root / "cache"
    cache.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "vicuna.rmt_span_selector.feature_manifest.v1",
                "feature_set_id": "fixture-core245",
                "features": [
                    {"feature_id": "1", "label": "evidence", "categories": ["evidence"], "causal_effect": 2.0},
                    {"feature_id": "2", "label": "format", "categories": ["format"], "causal_effect": 1.0},
                ],
                "sae_feature_ids": ["1", "2"],
            }
        ),
        encoding="utf-8",
    )
    _write_cache(cache, "q", "query", {"1": 1.0, "2": 0.0})
    _write_cache(cache, "p", "p", {"1": 1.0, "2": 0.0})
    _write_cache(cache, "n", "n", {"1": 0.0, "2": 1.0})
    groups = root / "groups.jsonl"
    groups.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query_activation_chunk_id": "q",
                "candidates": [
                    {"chunk_id": "p", "doc_id": "p", "label": 1, "dense_rank": 1, "dense_score": 0.9, "features": {"dense_score": 0.9}},
                    {"chunk_id": "n", "doc_id": "n", "label": 0, "dense_rank": 2, "dense_score": 0.8, "features": {"dense_score": 0.8}},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return groups, cache, manifest


def _write_cache(cache: Path, chunk_id: str, document_id: str, values: dict[str, float]) -> None:
    (cache / f"{chunk_id}.json").write_text(
        json.dumps({"chunk_id": chunk_id, "document_id": document_id, "telemetry_valid": True, "sae_feature_values": values}) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
