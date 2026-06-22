import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_behavior_latent_pilot import compare_pilot
from scripts.materialize_behavior_telemetry_cache import materialize_cache
from scripts.prepare_behavior_latent_reranker_pilot import build_split_groups
from scripts.run_resumable_behavior_capture import completed_chunk_ids, run_resumable_capture
from scripts.score_behavior_latent_reranker import score_checkpoint
from scripts.train_behavior_latent_reranker import transform_activation_values


class BehaviorLatentRerankerPilotTest(unittest.TestCase):
    def test_prepare_builds_pair_requests_and_behavior_ids(self):
        groups = [
            {
                "query_id": "q1",
                "query_text": "what treats lupus?",
                "candidates": [
                    {"chunk_id": "c1", "doc_id": "d1", "dense_rank": 1, "dense_score": 0.9, "label": 1, "text": "Hydroxychloroquine treats lupus."},
                    {"chunk_id": "c2", "doc_id": "d2", "dense_rank": 2, "dense_score": 0.8, "label": 0, "text": "This discusses unrelated treatment."},
                ],
            }
        ]

        prepared, requests = build_split_groups(
            dataset_name="fixture",
            split="train",
            source_groups=groups,
            limit=10,
            candidates_per_query=2,
            prompt_hash="abc",
            seed="13",
        )

        self.assertEqual(len(prepared), 1)
        self.assertEqual(len(requests), 2)
        self.assertIn("behavior_chunk_id", prepared[0]["candidates"][0])
        self.assertIn("Query:", requests[0]["prompt_text"])
        self.assertEqual(requests[0]["requested_prompt_section_label"], "query_candidate_behavior_prompt")

    def test_compare_pilot_reports_behavior_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups_path = root / "groups.jsonl"
            behavior_scores = root / "scores.jsonl"
            out = root / "summary.json"
            groups_path.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "query_text": "q",
                        "candidates": [
                            {"chunk_id": "c1", "doc_id": "d1", "dense_rank": 2, "dense_score": 0.1, "label": 1},
                            {"chunk_id": "c2", "doc_id": "d2", "dense_rank": 1, "dense_score": 0.2, "label": 0},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            behavior_scores.write_text(
                json.dumps({"query_id": "q1", "chunk_id": "c1", "score": 2.0}) + "\n"
                + json.dumps({"query_id": "q1", "chunk_id": "c2", "score": 1.0}) + "\n",
                encoding="utf-8",
            )

            summary = compare_pilot(
                groups_path=groups_path,
                behavior_scores_path=behavior_scores,
                out_path=out,
                actpred_scores_path=None,
                ettin_scores_path=None,
                randomization_iterations=10,
                top_k=10,
            )

            self.assertGreater(summary["paired"]["behavior_minus_dense"]["metric_deltas"]["ndcg@10"], 0.0)

    def test_materialize_cache_writes_per_chunk_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.jsonl"
            cache = root / "cache"
            capture.write_text(
                json.dumps({"chunk_id": "b1", "telemetry_valid": True, "sae_feature_values": {"1": 1.0}}) + "\n",
                encoding="utf-8",
            )

            summary = materialize_cache(capture_jsonl=capture, cache_dir=cache)

            self.assertEqual(summary["valid_count"], 1)
            self.assertTrue((cache / "b1.json").exists())

    def test_activation_transform_log1p_l2_normalizes_sparse_values(self):
        transformed = transform_activation_values([0.0, 3.0, 4.0], "log1p_l2")

        self.assertEqual(transformed[0], 0.0)
        self.assertAlmostEqual(sum(value * value for value in transformed), 1.0)
        self.assertGreater(transformed[2], transformed[1])

    def test_resumable_capture_plan_skips_cached_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests = root / "requests.jsonl"
            cache = root / "cache"
            cache.mkdir()
            requests.write_text(
                json.dumps({"chunk_id": "a"}) + "\n" + json.dumps({"chunk_id": "b"}) + "\n",
                encoding="utf-8",
            )
            (cache / "a.json").write_text(
                json.dumps({"chunk_id": "a", "telemetry_valid": True, "sae_feature_values": {"1": 1.0}}),
                encoding="utf-8",
            )

            summary = run_resumable_capture(
                requests_path=requests,
                output_jsonl=root / "rows.jsonl",
                cache_dir=cache,
                batch_dir=root / "batches",
                batch_size=1,
                max_batches=0,
                plan_only=True,
                capture_args=[],
            )

            self.assertEqual(completed_chunk_ids(cache), {"a"})
            self.assertEqual(summary["completed_before_count"], 1)
            self.assertEqual(summary["missing_before_count"], 1)
            self.assertEqual(summary["planned_batch_count"], 1)

    def test_score_checkpoint_scores_behavior_groups(self):
        torch = __import__("torch")
        nn = __import__("torch.nn", fromlist=["nn"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = root / "groups.jsonl"
            cache = root / "cache"
            checkpoint = root / "model.pt"
            cache.mkdir()
            groups.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "query_text": "q",
                        "candidates": [
                            {
                                "chunk_id": "c1",
                                "doc_id": "d1",
                                "dense_rank": 2,
                                "dense_score": 0.1,
                                "label": 1,
                                "behavior_chunk_id": "b1",
                                "features": {"dense_z": -1.0, "dense_rank_reciprocal": 0.5},
                            },
                            {
                                "chunk_id": "c2",
                                "doc_id": "d2",
                                "dense_rank": 1,
                                "dense_score": 0.2,
                                "label": 0,
                                "behavior_chunk_id": "b2",
                                "features": {"dense_z": 1.0, "dense_rank_reciprocal": 1.0},
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (cache / "b1.json").write_text(json.dumps({"chunk_id": "b1", "sae_feature_values": {"1": 3.0}}), encoding="utf-8")
            (cache / "b2.json").write_text(json.dumps({"chunk_id": "b2", "sae_feature_values": {"1": 0.0}}), encoding="utf-8")
            model = nn.Sequential(
                nn.Linear(4, 4),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(4, 4),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(4, 1),
            )
            torch.save(
                {
                    "feature_ids": ["1"],
                    "activation_transform": "raw",
                    "input_dim": 4,
                    "hidden_dim": 4,
                    "normalizer": {"mean": [0.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0, 1.0]},
                    "selected_alpha": 1.0,
                    "state_dict": model.state_dict(),
                    "prompt_representation": "query_candidate_pair_core245_max_prefill",
                },
                checkpoint,
            )

            summary = score_checkpoint(
                groups_path=groups,
                telemetry_cache_dir=cache,
                checkpoint_path=checkpoint,
                scores_out=root / "scores.jsonl",
                metrics_out=root / "metrics.json",
                top_k=10,
                device="cpu",
            )

            self.assertEqual(summary["scored_query_count"], 1)
            self.assertTrue((root / "scores.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
