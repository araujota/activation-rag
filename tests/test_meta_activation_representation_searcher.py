import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.train_meta_activation_representation_searcher import (
    RawExample,
    apply_geometry,
    fit_geometry,
    load_raw_examples,
    parse_source_spec,
    run_training,
)


class MetaActivationRepresentationSearcherTests(unittest.TestCase):
    def test_geometry_removes_train_centroid_and_preserves_dimension(self):
        examples = [
            RawExample("a", "q1", np.array([2.0, 0.0, 0.0]), np.array([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([1, 0]), np.array([1.0, 0.5])),
            RawExample("b", "q2", np.array([0.0, 2.0, 0.0]), np.array([[0.0, 3.0, 0.0], [1.0, 0.0, 0.0]]), np.array([1, 0]), np.array([1.0, 0.5])),
        ]

        geometry = fit_geometry(examples, geometry_policy="zscore_center_top_pc_l2", top_pc_removal=1, whitening_rank=0)
        transformed = apply_geometry(np.array([1.0, 1.0, 0.0]), geometry)

        self.assertEqual(geometry["output_dim"], 3)
        self.assertEqual(transformed.shape, (3,))
        self.assertEqual(len(geometry["components"]), 1)

    def test_run_training_writes_locked_manifest_without_loading_validation_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_manifest = root / "feature_manifest.json"
            feature_manifest.write_text(
                json.dumps(
                    {
                        "feature_set_id": "test-core",
                        "sae_feature_ids": ["1", "2", "3"],
                        "feature_metadata": {},
                    }
                ),
                encoding="utf-8",
            )
            train_groups = root / "train.jsonl"
            train_groups.write_text(json.dumps(_group("q1")) + "\n", encoding="utf-8")
            cache = root / "cache"
            cache.mkdir()
            _write_cache(cache, "query-q1", [1.0, 0.0, 0.0])
            _write_cache(cache, "pos-q1", [1.0, 0.2, 0.0])
            _write_cache(cache, "neg-q1", [0.0, 1.0, 0.0])
            locked_groups = root / "locked.jsonl"
            locked_groups.write_text("this is intentionally not json\n", encoding="utf-8")

            summary = run_training(
                train_sources=[parse_source_spec(f"toy:{train_groups}:{cache}")],
                locked_validation_sources=[parse_source_spec(f"locked:{locked_groups}:{cache}")],
                feature_manifest_path=feature_manifest,
                allow_positive_only_groups=False,
                representation="raw",
                geometry_policy="zscore_center_top_pc_l2",
                top_pc_removal=0,
                whitening_rank=0,
                model_out=root / "model.pt",
                metrics_out=root / "metrics.json",
                manifest_out=root / "locked-manifest.json",
                hidden_dim=8,
                dropout=0.0,
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                temperature=0.1,
                centroid_weight=0.1,
                inbatch_weight=0.0,
                margin_weight=0.0,
                uniformity_weight=0.0,
                hard_margin=0.1,
                hubness_sample_size=1,
                device="cpu",
                seed=3,
            )

            locked = json.loads((root / "locked-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "training_complete_no_heldout_validation_executed")
            self.assertEqual(locked["status"], "locked_not_executed")
            self.assertEqual(locked["sources"][0]["groups_line_count"], 1)
            self.assertTrue((root / "model.pt").exists())

    def test_positive_only_groups_require_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = root / "positive-only.jsonl"
            groups.write_text(json.dumps(_positive_only_group("q1")) + "\n", encoding="utf-8")
            cache = root / "cache"
            cache.mkdir()
            _write_cache(cache, "query-q1", [1.0, 0.0, 0.0])
            _write_cache(cache, "pos-q1", [1.0, 0.2, 0.0])
            source = parse_source_spec(f"toy:{groups}:{cache}")

            default_examples = load_raw_examples(
                [source],
                feature_ids=["1", "2", "3"],
                representation="raw",
                allow_positive_only_groups=False,
            )
            opt_in_examples = load_raw_examples(
                [source],
                feature_ids=["1", "2", "3"],
                representation="raw",
                allow_positive_only_groups=True,
            )

            self.assertEqual(default_examples, [])
            self.assertEqual(len(opt_in_examples), 1)
            self.assertEqual(opt_in_examples[0].labels.tolist(), [1])


def _group(query_id: str) -> dict:
    return {
        "query_id": query_id,
        "query_activation_chunk_id": f"query-{query_id}",
        "candidates": [
            {"chunk_id": f"pos-{query_id}", "doc_id": "pos", "label": 1, "dense_score": 0.9},
            {"chunk_id": f"neg-{query_id}", "doc_id": "neg", "label": 0, "dense_score": 0.8},
        ],
    }


def _positive_only_group(query_id: str) -> dict:
    return {
        "query_id": query_id,
        "query_activation_chunk_id": f"query-{query_id}",
        "candidates": [
            {"chunk_id": f"pos-{query_id}", "doc_id": "pos", "label": 1, "dense_score": 1.0},
        ],
    }


def _write_cache(cache: Path, chunk_id: str, values: list[float]) -> None:
    row = {
        "chunk_id": chunk_id,
        "capture_phase": "prefill",
        "generation_disabled": True,
        "prefill_only_extracted": True,
        "telemetry_valid": True,
        "sae_feature_values": {str(index + 1): value for index, value in enumerate(values)},
    }
    (cache / f"{chunk_id}.json").write_text(json.dumps(row), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
