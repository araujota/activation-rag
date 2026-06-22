from __future__ import annotations

import unittest

from activation_rag.schema import ChunkRecord, stable_hash
from scripts.prepare_techqa_rag_eval import select_positive_chunk_ids


def chunk(text: str, start: int, end: int, ordinal: int) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=stable_hash(f"{ordinal}:{text[start:end]}", 32),
        document_id="doc",
        ordinal=ordinal,
        text=text[start:end],
        text_hash=stable_hash(text[start:end], 32),
        char_start=start,
        char_end=end,
        token_count_estimate=1,
        chunker="test",
        chunk_size=100,
        chunk_overlap=0,
    )


class TechQAPositiveSelectionTests(unittest.TestCase):
    def test_selects_chunk_overlapping_normalized_answer_span(self) -> None:
        text = "Title\n\nFirst paragraph.\n\nThe workaround is to set the property directly on the instance.\n\nTail."
        chunks = [chunk(text, 0, 20, 0), chunk(text, 20, len(text), 1)]
        positives, policy = select_positive_chunk_ids(
            answer="The workaround is to set the property directly on the instance.",
            context_text=text,
            chunks=chunks,
        )
        self.assertEqual(policy, "answer_span_overlap")
        self.assertEqual(positives, {chunks[1].chunk_id})

    def test_falls_back_to_best_lexical_overlap(self) -> None:
        text = "Alpha install details.\n\nBeta SSL certificate handshake failure procedure."
        chunks = [chunk(text, 0, 23, 0), chunk(text, 23, len(text), 1)]
        positives, policy = select_positive_chunk_ids(
            answer="certificate handshake setup steps",
            context_text=text,
            chunks=chunks,
        )
        self.assertEqual(policy, "best_lexical_overlap_fallback")
        self.assertEqual(positives, {chunks[1].chunk_id})


if __name__ == "__main__":
    unittest.main()
