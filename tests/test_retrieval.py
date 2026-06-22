import unittest

from activation_rag.chunking import Chunker, ChunkerSettings
from activation_rag.embedding import HashEmbeddingProvider
from activation_rag.pipeline import RagEngine
from activation_rag.retrieval import ActivationMatchingConfig, rank_activation, rank_activation_with_strategy
from activation_rag.schema import ActivationRecord, DocumentRecord, RetrievalResult
from activation_rag.telemetry import MockTelemetryProvider


class RetrievalTests(unittest.TestCase):
    def test_activation_rank_skips_mismatched_vector_dimensions(self):
        query = activation_record("query", {"a": 1.0})
        good = activation_record("good", {"a": 0.9})
        mismatched = activation_record("bad", {"a": 1.0, "b": 1.0})

        results = rank_activation(query, [mismatched, good], top_k=2)

        self.assertEqual([result.chunk_id for result in results], ["good"])

    def test_activation_strategy_default_matches_raw_cosine(self):
        query = activation_record("query", {"a": 1.0, "b": 0.1})
        records = [
            activation_record("weaker", {"a": 0.7, "b": 0.7}),
            activation_record("stronger", {"a": 1.0, "b": 0.2}),
        ]

        raw = rank_activation(query, records, top_k=2)
        configured = rank_activation_with_strategy(query, records, top_k=2)

        self.assertEqual([result.chunk_id for result in configured], [result.chunk_id for result in raw])
        self.assertEqual([result.strategy for result in configured], ["activation-sim", "activation-sim"])

    def test_csls_activation_strategy_penalizes_dense_hubs(self):
        query = activation_record("query", {"x": 1.0, "y": 0.12, "z": 0.0})
        records = [
            activation_record("hub", {"x": 1.0, "y": 0.0, "z": 0.0}),
            activation_record("hub-neighbor-a", {"x": 0.99, "y": 0.02, "z": 0.0}),
            activation_record("hub-neighbor-b", {"x": 0.98, "y": -0.04, "z": 0.0}),
            activation_record("hub-neighbor-c", {"x": 0.97, "y": 0.01, "z": 0.0}),
            activation_record("specific", {"x": 0.92, "y": 0.39, "z": 0.0}),
        ]

        raw = rank_activation(query, records, top_k=1)
        corrected = rank_activation_with_strategy(
            query,
            records,
            top_k=1,
            config=ActivationMatchingConfig(strategy="csls", local_k=2),
        )

        self.assertEqual(raw[0].chunk_id, "hub-neighbor-a")
        self.assertEqual(corrected[0].chunk_id, "specific")
        self.assertEqual(corrected[0].strategy, "activation-csls")

    def test_nicdm_activation_strategy_uses_local_distance_scaling(self):
        query = activation_record("query", {"x": 1.0, "y": 0.12, "z": 0.0})
        records = [
            activation_record("hub", {"x": 1.0, "y": 0.0, "z": 0.0}),
            activation_record("hub-neighbor-a", {"x": 0.99, "y": 0.02, "z": 0.0}),
            activation_record("hub-neighbor-b", {"x": 0.98, "y": -0.04, "z": 0.0}),
            activation_record("hub-neighbor-c", {"x": 0.97, "y": 0.01, "z": 0.0}),
            activation_record("specific", {"x": 0.92, "y": 0.39, "z": 0.0}),
        ]

        corrected = rank_activation_with_strategy(
            query,
            records,
            top_k=1,
            config=ActivationMatchingConfig(strategy="nicdm", local_k=2),
        )

        self.assertEqual(corrected[0].chunk_id, "specific")
        self.assertEqual(corrected[0].strategy, "activation-nicdm")

    def test_whiten_l2_activation_strategy_downweights_high_variance_axes(self):
        query = activation_record("query", {"x": 0.0, "y": 3.0, "z": 1.0})
        records = [
            activation_record("high-var-positive", {"x": 0.0, "y": 10.0, "z": 0.0}),
            activation_record("high-var-negative", {"x": 0.0, "y": -10.0, "z": 0.0}),
            activation_record("high-var-positive-2", {"x": 0.0, "y": 9.0, "z": 0.0}),
            activation_record("high-var-negative-2", {"x": 0.0, "y": -9.0, "z": 0.0}),
            activation_record("specific", {"x": 0.0, "y": 0.0, "z": 1.0}),
        ]

        raw = rank_activation(query, records, top_k=1)
        corrected = rank_activation_with_strategy(
            query,
            records,
            top_k=1,
            config=ActivationMatchingConfig(strategy="whiten_l2", whiten_dimensions=3),
        )

        self.assertEqual(raw[0].chunk_id, "high-var-positive")
        self.assertEqual(corrected[0].chunk_id, "specific")

    def test_top_pc_removed_activation_strategy_removes_common_dominating_direction(self):
        query = activation_record("query", {"x": 0.0, "y": 3.0, "z": 1.0})
        records = [
            activation_record("dominant-positive", {"x": 0.0, "y": 10.0, "z": 0.0}),
            activation_record("dominant-negative", {"x": 0.0, "y": -10.0, "z": 0.0}),
            activation_record("dominant-positive-2", {"x": 0.0, "y": 9.0, "z": 0.0}),
            activation_record("dominant-negative-2", {"x": 0.0, "y": -9.0, "z": 0.0}),
            activation_record("specific", {"x": 0.0, "y": 0.0, "z": 1.0}),
        ]

        corrected = rank_activation_with_strategy(
            query,
            records,
            top_k=1,
            config=ActivationMatchingConfig(strategy="top_pc_removed", remove_components=1),
        )

        self.assertEqual(corrected[0].chunk_id, "specific")
        self.assertEqual(corrected[0].strategy, "activation-top-pc-removed")

    def test_per_site_late_fusion_can_weight_discriminative_sites(self):
        query = activation_record(
            "query",
            {
                "act:site_good:prefill_last:mean": 1.0,
                "act:site_good:prefill_last:max": 0.0,
                "act:site_bad:prefill_last:mean": 0.0,
            },
        )
        records = [
            activation_record(
                "bad-site-match",
                {
                    "act:site_good:prefill_last:mean": 0.0,
                    "act:site_good:prefill_last:max": 1.0,
                    "act:site_bad:prefill_last:mean": 1.0,
                },
            ),
            activation_record(
                "good-site-match",
                {
                    "act:site_good:prefill_last:mean": 1.0,
                    "act:site_good:prefill_last:max": 0.0,
                    "act:site_bad:prefill_last:mean": -1.0,
                },
            ),
        ]

        corrected = rank_activation_with_strategy(
            query,
            records,
            top_k=1,
            config=ActivationMatchingConfig(
                strategy="per_site_late_fusion",
                site_weights={"site_good": 1.0, "site_bad": 0.0},
            ),
        )

        self.assertEqual(corrected[0].chunk_id, "good-site-match")
        self.assertIn("activation-site:site_good", corrected[0].component_scores)

    def test_activation_similarity_can_rank_without_dense_embeddings(self):
        docs = [
            DocumentRecord.from_text(
                source_uri="memory://a",
                title="Verification",
                text="Verification pressure rises when evidence must be checked carefully.",
            ),
            DocumentRecord.from_text(
                source_uri="memory://b",
                title="Termination",
                text="Termination readiness rises when the answer is already concise.",
            ),
        ]
        engine = RagEngine(
            chunker=Chunker(ChunkerSettings(chunk_size=30, chunk_overlap=0)),
            embedder=HashEmbeddingProvider(dimension=16),
            telemetry_provider=MockTelemetryProvider(),
        )
        engine.ingest(docs, embed=False, capture_telemetry=True)

        results = engine.search_activation("verification evidence", top_k=2)

        self.assertEqual(results[0].strategy, "activation-sim")
        self.assertIn("Verification", engine.get_chunk(results[0].chunk_id).text)
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_dense_search_works_independently_from_telemetry(self):
        docs = [
            DocumentRecord.from_text("memory://dense-a", "Dense A", "alpha beta gamma"),
            DocumentRecord.from_text("memory://dense-b", "Dense B", "repair verification evidence"),
        ]
        engine = RagEngine(
            chunker=Chunker(ChunkerSettings(chunk_size=20, chunk_overlap=0)),
            embedder=HashEmbeddingProvider(dimension=32),
            telemetry_provider=MockTelemetryProvider(),
        )
        engine.ingest(docs, embed=True, capture_telemetry=False)

        results = engine.search_dense("verification evidence", top_k=1)

        self.assertEqual(results[0].strategy, "dense")
        self.assertIn("verification", engine.get_chunk(results[0].chunk_id).text)

    def test_activation_rerank_reorders_dense_candidate_pool_only(self):
        docs = [
            DocumentRecord.from_text("memory://a", "Dense Distractor", "verification token alpha alpha"),
            DocumentRecord.from_text("memory://b", "Activation Match", "evidence should be checked carefully"),
            DocumentRecord.from_text("memory://c", "Outside Pool", "termination answer concise final"),
        ]
        engine = RagEngine(
            chunker=Chunker(ChunkerSettings(chunk_size=20, chunk_overlap=0)),
            embedder=HashEmbeddingProvider(dimension=32),
            telemetry_provider=MockTelemetryProvider(),
        )
        chunks = engine.ingest(docs, embed=True, capture_telemetry=True)
        dense_candidates = [
            # Deliberately put the weaker activation match first to prove reranking reorders.
            # The third chunk is omitted to prove reranking does not broaden the candidate pool.
            RetrievalResult(chunks[0].chunk_id, "dense", 0.90, 1, {"dense": 0.90}),
            RetrievalResult(chunks[1].chunk_id, "dense", 0.10, 2, {"dense": 0.10}),
        ]

        reranked = engine.rerank_with_activation("verification evidence", dense_candidates, top_k=2)

        self.assertEqual(reranked[0].strategy, "dense+activation-rerank")
        self.assertEqual(reranked[0].chunk_id, chunks[1].chunk_id)
        self.assertEqual({result.chunk_id for result in reranked}, {chunks[0].chunk_id, chunks[1].chunk_id})
        self.assertNotIn(chunks[2].chunk_id, {result.chunk_id for result in reranked})
        self.assertIn("activation-sim", reranked[0].component_scores)
        self.assertIn("dense", reranked[0].component_scores)

    def test_compare_search_methods_returns_all_result_groups(self):
        docs = [
            DocumentRecord.from_text("memory://verify", "Verify", "verification evidence checked carefully"),
            DocumentRecord.from_text("memory://plan", "Plan", "plan architecture steps sequencing"),
            DocumentRecord.from_text("memory://final", "Final", "termination answer concise final"),
        ]
        engine = RagEngine(
            chunker=Chunker(ChunkerSettings(chunk_size=20, chunk_overlap=0)),
            embedder=HashEmbeddingProvider(dimension=32),
            telemetry_provider=MockTelemetryProvider(),
        )
        engine.ingest(docs, embed=True, capture_telemetry=True)

        comparison = engine.compare_search_methods("verification evidence", top_k=2, candidate_k=3)

        self.assertEqual(comparison.query, "verification evidence")
        self.assertEqual(len(comparison.dense_results), 2)
        self.assertEqual(len(comparison.activation_results), 2)
        self.assertEqual(len(comparison.activation_reranked_results), 2)
        self.assertEqual(comparison.activation_results[0].strategy, "activation-sim")
        self.assertEqual(comparison.activation_reranked_results[0].strategy, "dense+activation-rerank")


if __name__ == "__main__":
    unittest.main()


def activation_record(chunk_id, features):
    return ActivationRecord(
        chunk_id=chunk_id,
        document_id="doc",
        capture_run_id="run",
        provider_id="provider",
        model_id="model",
        site_id="site",
        current_em_state={},
        neutral_baseline_state={},
        prior_current_state={},
        delta_vs_neutral={},
        delta_vs_current={},
        saturation={},
        residual_headroom={},
        positive_mass=0.0,
        negative_mass=0.0,
        total_mass=0.0,
        signed_balance=0.0,
        sae_feature_values=features,
        sae_delta_vs_neutral=features,
        sae_delta_vs_current=features,
        sae_feature_mask={key: True for key in features},
    )
