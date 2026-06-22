import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.capture_beir_telemetry_cache import run_capture


class CaptureBeirTelemetryCacheTests(unittest.TestCase):
    def test_run_capture_batches_beir_chunks_and_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            beir = root / "beir"
            cache = root / "cache"
            beir.mkdir()
            (beir / "qrels").mkdir()
            (beir / "corpus.jsonl").write_text(
                json.dumps({"_id": "d1", "title": "Doc", "text": "alpha positive evidence"}) + "\n",
                encoding="utf-8",
            )
            (beir / "queries.jsonl").write_text(
                json.dumps({"_id": "q1", "text": "alpha query"}) + "\n",
                encoding="utf-8",
            )
            (beir / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")
            fake = root / "fake_capture.py"
            fake.write_text(_fake_capture_code(), encoding="utf-8")

            summary = run_capture(
                beir_dir=beir,
                dataset_name="toy",
                split="test",
                telemetry_command=(sys.executable, str(fake), "{input_jsonl}", "{output_jsonl}"),
                cache_dir=cache,
                batch_size=1,
                include_documents=True,
                include_queries=True,
                limit=None,
                provider_id="fake-prefill",
                model_id="fake-model",
                site_id="fake-site",
                layer_selection_policy="fake-policy",
                prompt_template_id="fake-template",
                normalization_policy="fake-normalization",
                timeout_seconds=30,
            )

            self.assertEqual(summary["captured_count"], 2)
            self.assertEqual(summary["batch_count"], 2)
            self.assertEqual(summary["uncached_count"], 2)
            self.assertEqual(summary["attempted_uncached_count"], 2)
            self.assertEqual(len(list(cache.glob("*.json"))), 2)

    def test_run_capture_reports_true_cache_hits_when_limited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            beir = root / "beir"
            cache = root / "cache"
            beir.mkdir()
            (beir / "qrels").mkdir()
            (beir / "corpus.jsonl").write_text(
                json.dumps({"_id": "d1", "title": "Doc", "text": "alpha positive evidence"}) + "\n"
                + json.dumps({"_id": "d2", "title": "Doc", "text": "beta positive evidence"}) + "\n",
                encoding="utf-8",
            )
            (beir / "queries.jsonl").write_text(
                json.dumps({"_id": "q1", "text": "alpha query"}) + "\n",
                encoding="utf-8",
            )
            (beir / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")
            fake = root / "fake_capture.py"
            fake.write_text(_fake_capture_code(), encoding="utf-8")

            summary = run_capture(
                beir_dir=beir,
                dataset_name="toy",
                split="test",
                telemetry_command=(sys.executable, str(fake), "{input_jsonl}", "{output_jsonl}"),
                cache_dir=cache,
                batch_size=1,
                include_documents=True,
                include_queries=False,
                limit=1,
                provider_id="fake-prefill",
                model_id="fake-model",
                site_id="fake-site",
                layer_selection_policy="fake-policy",
                prompt_template_id="fake-template",
                normalization_policy="fake-normalization",
                timeout_seconds=30,
            )

            self.assertEqual(summary["candidate_count"], 2)
            self.assertEqual(summary["cache_hit_count"], 0)
            self.assertEqual(summary["uncached_count"], 2)
            self.assertEqual(summary["attempted_uncached_count"], 1)
            self.assertEqual(summary["captured_count"], 1)


def _fake_capture_code() -> str:
    return """
import json
import sys
from pathlib import Path

inp = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for line in inp.read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    rows.append({
        'chunk_id': row['chunk_id'],
        'document_id': row['document_id'],
        'capture_run_id': row['capture_run_id'],
        'provider_id': row['provider_id'],
        'model_id': row['model_id'],
        'site_id': row['site_id'],
        'layer_selection_policy': row['layer_selection_policy'],
        'prompt_template_id': row['prompt_template_id'],
        'prompt_template_hash': row['prompt_template_hash'],
        'normalization_policy': row['normalization_policy'],
        'capture_phase': 'prefill',
        'generation_disabled': True,
        'prefill_only_extracted': True,
        'sae_feature_values': {'act:site:prefill_last:mean': 1.0},
        'sae_delta_vs_neutral': {'act:site:prefill_last:mean': 1.0},
        'sae_delta_vs_current': {'act:site:prefill_last:mean': 1.0},
        'sae_feature_mask': {'act:site:prefill_last:mean': True},
        'telemetry_valid': True,
        'provenance': {'capture_phase': 'prefill'},
    })
out.write_text(''.join(json.dumps(row) + '\\n' for row in rows), encoding='utf-8')
"""


if __name__ == "__main__":
    unittest.main()
