from activation_rag import Chunker, ChunkerSettings, DocumentRecord, HashEmbeddingProvider, MockTelemetryProvider, RagEngine


def build_engine() -> RagEngine:
    docs = [
        DocumentRecord.from_text(
            "example://incident-review",
            "Incident Review",
            "The deployment failed after a configuration change. The repair plan is to verify the failing health check, inspect the logs, and fix the bad environment variable before restarting the worker. Evidence should be checked carefully before finalizing the incident report.",
        ),
        DocumentRecord.from_text(
            "example://travel-policy",
            "Travel Policy",
            "Employees should book travel through the approved portal. If a destination is uncertain, ask a clarifying question before booking. A concise final answer is enough when the itinerary is already confirmed.",
        ),
        DocumentRecord.from_text(
            "example://architecture-brief",
            "Architecture Brief",
            "The new retrieval service needs a plan with clear sequencing. First define the document contract, then build the chunk store, then add dense search and activation similarity as separate strategies.",
        ),
    ]
    engine = RagEngine(
        chunker=Chunker(ChunkerSettings(chunk_size=18, chunk_overlap=3)),
        embedder=HashEmbeddingProvider(dimension=64),
        telemetry_provider=MockTelemetryProvider(),
    )
    engine.ingest(docs, embed=True, capture_telemetry=True)
    return engine


def print_group(engine: RagEngine, label: str, results) -> None:
    print(label)
    for result in results:
        chunk = engine.get_chunk(result.chunk_id)
        scores = ", ".join(f"{key}={value:.3f}" for key, value in sorted(result.component_scores.items()))
        print(f"  {result.rank}. score={result.score:.3f} [{scores}] {chunk.text}")
    print()


def main() -> None:
    query = "verification evidence health check"
    engine = build_engine()
    comparison = engine.compare_search_methods(query, top_k=3, candidate_k=5)

    print(f"Query: {comparison.query}")
    print(f"Dense/activation overlap: {len(comparison.dense_activation_overlap)}")
    print(f"Dense/rerank overlap: {len(comparison.dense_rerank_overlap)}")
    print()
    print_group(engine, "Traditional Dense RAG", comparison.dense_results)
    print_group(engine, "Activation KNN", comparison.activation_results)
    print_group(engine, "Dense RAG + Activation Rerank", comparison.activation_reranked_results)
    for note in comparison.notes:
        print(f"Note: {note}")


if __name__ == "__main__":
    main()

