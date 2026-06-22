import unittest
import json
import sys
import tempfile
from pathlib import Path

from activation_rag.chunking import Chunker, ChunkerSettings
from activation_rag.embedding import CommandEmbeddingProvider, SentenceTransformerEmbeddingProvider
from activation_rag.schema import DocumentRecord


class FakeSentenceTransformer:
    def __init__(self):
        self.calls = []

    def encode(self, texts, *, batch_size, normalize_embeddings, show_progress_bar):
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
            }
        )
        return [[1.0, 0.0], [0.0, 1.0]][: len(texts)]


class SentenceTransformerEmbeddingProviderTests(unittest.TestCase):
    def test_embeds_chunks_with_real_model_id_and_batch_settings(self):
        docs = [
            DocumentRecord.from_text("memory://a", "A", "alpha evidence"),
            DocumentRecord.from_text("memory://b", "B", "beta travel"),
        ]
        chunks = Chunker(ChunkerSettings(chunk_size=20, chunk_overlap=0)).split(docs)
        fake_model = FakeSentenceTransformer()
        provider = SentenceTransformerEmbeddingProvider(
            model_name="BAAI/bge-base-en-v1.5",
            batch_size=8,
            model=fake_model,
        )

        records = provider.embed_chunks(chunks)

        self.assertEqual(provider.model_id, "sentence-transformers:BAAI/bge-base-en-v1.5")
        self.assertEqual([record.model_id for record in records], [provider.model_id, provider.model_id])
        self.assertEqual(records[0].vector, (1.0, 0.0))
        self.assertEqual(records[1].vector, (0.0, 1.0))
        self.assertEqual(fake_model.calls[0]["batch_size"], 8)
        self.assertTrue(fake_model.calls[0]["normalize_embeddings"])
        self.assertFalse(fake_model.calls[0]["show_progress_bar"])

    def test_command_embedding_provider_reads_vectors_from_external_command(self):
        docs = [
            DocumentRecord.from_text("memory://a", "A", "alpha evidence"),
            DocumentRecord.from_text("memory://b", "B", "beta travel"),
        ]
        chunks = Chunker(ChunkerSettings(chunk_size=20, chunk_overlap=0)).split(docs)
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake_embed.py"
            script.write_text(
                """
import json
import sys

input_path, output_path = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in open(input_path, encoding="utf-8") if line.strip()]
with open(output_path, "w", encoding="utf-8") as sink:
    for index, row in enumerate(rows):
        vector = [1.0, 0.0] if index == 0 else [0.0, 1.0]
        sink.write(json.dumps({"id": row["id"], "vector": vector}) + "\\n")
""".lstrip(),
                encoding="utf-8",
            )
            provider = CommandEmbeddingProvider(
                command=[sys.executable, str(script), "{input_jsonl}", "{output_jsonl}"],
                model_id="command:bge-base-en-v1.5",
            )

            records = provider.embed_chunks(chunks)

        self.assertEqual([record.chunk_id for record in records], [chunk.chunk_id for chunk in chunks])
        self.assertEqual(records[0].model_id, "command:bge-base-en-v1.5")
        self.assertEqual(records[0].vector, (1.0, 0.0))
        self.assertEqual(records[1].vector, (0.0, 1.0))

    def test_command_embedding_provider_retries_transient_resource_failure(self):
        docs = [DocumentRecord.from_text("memory://a", "A", "alpha evidence")]
        chunks = Chunker(ChunkerSettings(chunk_size=20, chunk_overlap=0)).split(docs)
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "flaky_embed.py"
            counter = Path(tmp) / "attempts.txt"
            script.write_text(
                """
import json
import sys
from pathlib import Path

input_path, output_path, counter_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
attempt = int(counter_path.read_text()) + 1 if counter_path.exists() else 1
counter_path.write_text(str(attempt))
if attempt == 1:
    print("BlockingIOError: [Errno 35] Resource temporarily unavailable", file=sys.stderr)
    raise SystemExit(1)
rows = [json.loads(line) for line in open(input_path, encoding="utf-8") if line.strip()]
with open(output_path, "w", encoding="utf-8") as sink:
    for row in rows:
        sink.write(json.dumps({"id": row["id"], "vector": [1.0, 0.0]}) + "\\n")
""".lstrip(),
                encoding="utf-8",
            )
            provider = CommandEmbeddingProvider(
                command=[sys.executable, str(script), "{input_jsonl}", "{output_jsonl}", str(counter)],
                model_id="command:flaky",
                max_attempts=3,
                retry_backoff_seconds=0.0,
            )

            records = provider.embed_chunks(chunks)
            self.assertEqual(counter.read_text(), "2")

        self.assertEqual(records[0].vector, (1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
