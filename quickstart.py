from activation_rag import Chunker, ChunkerSettings, DocumentRecord, HashEmbeddingProvider, MockTelemetryProvider, RagEngine


def main() -> None:
    docs = [
        DocumentRecord.from_text(
            "memory://verification",
            "Verification Notes",
            "Verification pressure rises when evidence must be checked carefully before finalizing an answer.",
        ),
        DocumentRecord.from_text(
            "memory://planning",
            "Planning Notes",
            "Plan formation rises when a task needs architecture, steps, and sequencing.",
        ),
    ]
    engine = RagEngine(
        chunker=Chunker(ChunkerSettings(chunk_size=32, chunk_overlap=4)),
        embedder=HashEmbeddingProvider(dimension=64),
        telemetry_provider=MockTelemetryProvider(),
    )
    engine.ingest(docs)

    print("Dense:")
    for result in engine.search_dense("verification evidence", top_k=2):
        print(result.rank, result.score, engine.get_chunk(result.chunk_id).text)

    print("\nActivation:")
    for result in engine.search_activation("verification evidence", top_k=2):
        print(result.rank, result.score, engine.get_chunk(result.chunk_id).text)


if __name__ == "__main__":
    main()

