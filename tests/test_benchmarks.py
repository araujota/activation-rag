import unittest
import json
import tempfile
from pathlib import Path

from activation_rag.benchmarks import (
    BenchmarkDataset,
    assert_benchmark_telemetry_allowed,
    evaluate_dataset,
    load_beir_dataset,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from activation_rag.chunking import Chunker, ChunkerSettings
from activation_rag.embedding import HashEmbeddingProvider
from activation_rag.retrieval import ActivationMatchingConfig
from activation_rag.telemetry import MockTelemetryProvider


class CountingTelemetryProvider:
    provider_id = "counting-real-prefill"
    model_id = "counting-model"

    def __init__(self):
        self.inner = MockTelemetryProvider(provider_id=self.provider_id, model_id=self.model_id)
        self.batch_sizes = []

    def capture_prefill(self, chunks):
        self.batch_sizes.append(len(chunks))
        return self.inner.capture_prefill(chunks)


class CountingEmbeddingProvider:
    model_id = "counting-embedding"

    def __init__(self):
        self.inner = HashEmbeddingProvider(dimension=64, model_id=self.model_id)
        self.query_batches = []

    def embed_chunks(self, chunks):
        return self.inner.embed_chunks(chunks)

    def embed_texts(self, texts):
        self.query_batches.append(list(texts))
        return self.inner.embed_texts(texts)


class BenchmarkMetricTests(unittest.TestCase):
    def test_metrics_match_known_values(self):
        ranked = ["d3", "d2", "d1", "d4"]
        qrels = {"d1": 1, "d2": 1}

        self.assertAlmostEqual(mean_reciprocal_rank(ranked, qrels, 10), 0.5)
        self.assertAlmostEqual(recall_at_k(ranked, qrels, 1), 0.0)
        self.assertAlmostEqual(recall_at_k(ranked, qrels, 2), 0.5)
        self.assertAlmostEqual(ndcg_at_k(ranked, qrels, 2), 0.6309297535714575 / 1.6309297535714575)

    def test_fixture_evaluation_returns_three_approaches(self):
        dataset = BenchmarkDataset(
            name="fixture",
            split="test",
            corpus={
                "doc-evidence": "Evidence should be checked carefully before finalizing the incident report.",
                "doc-health": "The repair plan is to verify the failing health check and inspect logs.",
                "doc-travel": "Employees should book travel through the approved portal.",
            },
            queries={"q1": "verification evidence health check"},
            qrels={"q1": {"doc-evidence": 1, "doc-health": 1}},
            metric_profile="beir",
        )

        summary = evaluate_dataset(
            dataset,
            chunker=Chunker(ChunkerSettings(chunk_size=64, chunk_overlap=0)),
            embedder=HashEmbeddingProvider(dimension=64),
            telemetry_provider=MockTelemetryProvider(),
            top_k=3,
            candidate_k=3,
        )

        self.assertEqual(summary.dataset_name, "fixture")
        self.assertEqual(summary.query_count, 1)
        self.assertEqual(set(summary.metrics_by_approach), {"dense", "activation-sim", "dense+activation-rerank"})
        self.assertGreater(summary.metrics_by_approach["dense"]["recall@3"], 0.0)
        self.assertGreater(summary.metrics_by_approach["activation-sim"]["recall@3"], 0.0)
        self.assertGreater(summary.metrics_by_approach["dense+activation-rerank"]["recall@3"], 0.0)

    def test_fixture_evaluation_can_use_explicit_activation_matching_strategy(self):
        dataset = BenchmarkDataset(
            name="fixture-csls",
            split="test",
            corpus={
                "doc-evidence": "Evidence should be checked carefully before finalizing the incident report.",
                "doc-health": "The repair plan is to verify the failing health check and inspect logs.",
                "doc-travel": "Employees should book travel through the approved portal.",
            },
            queries={"q1": "verification evidence health check"},
            qrels={"q1": {"doc-evidence": 1, "doc-health": 1}},
            metric_profile="beir",
        )

        summary = evaluate_dataset(
            dataset,
            chunker=Chunker(ChunkerSettings(chunk_size=64, chunk_overlap=0)),
            embedder=HashEmbeddingProvider(dimension=64),
            telemetry_provider=MockTelemetryProvider(),
            top_k=3,
            candidate_k=3,
            activation_matching_config=ActivationMatchingConfig(strategy="csls", local_k=1),
        )

        self.assertEqual(
            set(summary.metrics_by_approach),
            {"dense", "activation-csls", "dense+activation-csls-rerank"},
        )
        self.assertIn("activation_strategy=csls", summary.notes)

    def test_nonfixture_benchmark_rejects_mock_telemetry_without_explicit_override(self):
        provider = MockTelemetryProvider()

        with self.assertRaisesRegex(ValueError, "mock telemetry"):
            assert_benchmark_telemetry_allowed(provider, fixture=False, allow_mock_telemetry=False)

        assert_benchmark_telemetry_allowed(provider, fixture=True, allow_mock_telemetry=False)
        assert_benchmark_telemetry_allowed(provider, fixture=False, allow_mock_telemetry=True)

    def test_multichunk_documents_preserve_benchmark_doc_id_mapping(self):
        dataset = BenchmarkDataset(
            name="fixture-long",
            split="test",
            corpus={
                "doc-long": "verification evidence " * 40,
                "doc-other": "travel portal booking " * 20,
            },
            queries={"q1": "verification evidence"},
            qrels={"q1": {"doc-long": 1}},
            metric_profile="beir",
        )

        summary = evaluate_dataset(
            dataset,
            chunker=Chunker(ChunkerSettings(chunk_size=8, chunk_overlap=0)),
            embedder=HashEmbeddingProvider(dimension=64),
            telemetry_provider=MockTelemetryProvider(),
            top_k=3,
            candidate_k=3,
        )

        self.assertEqual(summary.metrics_by_approach["dense"]["recall@3"], 1.0)

    def test_evaluation_captures_query_prefill_once_per_query(self):
        dataset = BenchmarkDataset(
            name="fixture-query-capture",
            split="test",
            corpus={
                "doc-evidence": "verification evidence " * 4,
                "doc-other": "travel portal " * 4,
            },
            queries={"q1": "verification evidence", "q2": "travel portal"},
            qrels={"q1": {"doc-evidence": 1}, "q2": {"doc-other": 1}},
            metric_profile="beir",
        )
        telemetry_provider = CountingTelemetryProvider()

        evaluate_dataset(
            dataset,
            chunker=Chunker(ChunkerSettings(chunk_size=32, chunk_overlap=0)),
            embedder=HashEmbeddingProvider(dimension=64),
            telemetry_provider=telemetry_provider,
            top_k=2,
            candidate_k=2,
        )

        self.assertEqual(telemetry_provider.batch_sizes[0], 2)
        self.assertEqual(telemetry_provider.batch_sizes[1:], [1, 1])

    def test_evaluation_batches_dense_query_embeddings(self):
        dataset = BenchmarkDataset(
            name="fixture-query-embeddings",
            split="test",
            corpus={
                "doc-evidence": "verification evidence " * 4,
                "doc-other": "travel portal " * 4,
            },
            queries={"q1": "verification evidence", "q2": "travel portal"},
            qrels={"q1": {"doc-evidence": 1}, "q2": {"doc-other": 1}},
            metric_profile="beir",
        )
        embedder = CountingEmbeddingProvider()

        evaluate_dataset(
            dataset,
            chunker=Chunker(ChunkerSettings(chunk_size=32, chunk_overlap=0)),
            embedder=embedder,
            telemetry_provider=MockTelemetryProvider(),
            top_k=2,
            candidate_k=2,
        )

        self.assertEqual(embedder.query_batches, [["verification evidence", "travel portal"]])

    def test_load_beir_dataset_from_jsonl_and_qrels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "toy"
            (root / "qrels").mkdir(parents=True)
            (root / "corpus.jsonl").write_text(
                json.dumps({"_id": "d1", "title": "Title", "text": "verification evidence", "metadata": {}}) + "\n",
                encoding="utf-8",
            )
            (root / "queries.jsonl").write_text(
                json.dumps({"_id": "q1", "text": "verification"}) + "\n"
                + json.dumps({"_id": "q-unjudged", "text": "unjudged"}) + "\n",
                encoding="utf-8",
            )
            (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")

            dataset = load_beir_dataset(root, name="toy", split="test")

        self.assertEqual(dataset.corpus["d1"], "Title\n\nverification evidence")
        self.assertEqual(dataset.queries, {"q1": "verification"})
        self.assertEqual(dataset.qrels["q1"], {"d1": 1})
        self.assertEqual(dataset.metric_profile, "beir")


if __name__ == "__main__":
    unittest.main()
