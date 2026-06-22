# Quickstart: Activation Telemetry RAG

After implementation:

```bash
python -m unittest discover -s tests
python quickstart.py
```

Expected quickstart behavior:

1. Build two sample documents.
2. Split them into stable chunks.
3. Embed chunks with the deterministic hash embedder.
4. Capture selector-compatible mock telemetry for each chunk.
5. Query with dense retrieval.
6. Query with activation-similarity retrieval.

