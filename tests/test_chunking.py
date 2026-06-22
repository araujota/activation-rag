import unittest

from activation_rag.chunking import Chunker, ChunkerSettings
from activation_rag.schema import DocumentRecord


class ChunkingTests(unittest.TestCase):
    def test_stable_chunk_ids_for_same_document_and_settings(self):
        doc = DocumentRecord.from_text(
            source_uri="memory://doc-a",
            title="Doc A",
            text="Alpha sentence one. Alpha sentence two.\n\nBeta sentence three.",
        )
        settings = ChunkerSettings(chunk_size=8, chunk_overlap=2)
        chunker = Chunker(settings)

        first = chunker.split([doc])
        second = chunker.split([doc])

        self.assertGreater(len(first), 1)
        self.assertEqual([chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second])
        self.assertEqual([chunk.text for chunk in first], [chunk.text for chunk in second])

    def test_chunks_preserve_offsets_and_text_hashes(self):
        text = "One two three four five. Six seven eight nine ten."
        doc = DocumentRecord.from_text(source_uri="memory://doc-b", title="Doc B", text=text)
        chunks = Chunker(ChunkerSettings(chunk_size=6, chunk_overlap=1)).split([doc])

        for chunk in chunks:
            self.assertEqual(text[chunk.char_start : chunk.char_end], chunk.text)
            self.assertEqual(chunk.text_hash, DocumentRecord.hash_text(chunk.text))
            self.assertLess(chunk.char_start, chunk.char_end)


if __name__ == "__main__":
    unittest.main()

