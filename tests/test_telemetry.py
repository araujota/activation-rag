import json
import sys
import tempfile
import unittest
from pathlib import Path

from activation_rag.chunking import Chunker, ChunkerSettings
from activation_rag.schema import DocumentRecord
from activation_rag.embedding import HashEmbeddingProvider
from activation_rag.pipeline import RagEngine
from activation_rag.telemetry import CommandPrefillTelemetryProvider, MockTelemetryProvider


class TelemetryTests(unittest.TestCase):
    def test_mock_provider_emits_selector_compatible_delta_fields(self):
        doc = DocumentRecord.from_text(
            source_uri="memory://telemetry",
            title="Telemetry",
            text="Verify the evidence carefully before finalizing the answer.",
        )
        chunk = Chunker(ChunkerSettings(chunk_size=20, chunk_overlap=0)).split([doc])[0]
        provider = MockTelemetryProvider(
            neutral_baseline_state={"verification_pressure": 0.1, "plan_formation": 0.2},
            prior_current_state={"verification_pressure": 0.3, "plan_formation": 0.1},
        )

        record = provider.capture_prefill([chunk])[0]

        self.assertEqual(record.chunk_id, chunk.chunk_id)
        self.assertTrue(record.telemetry_valid)
        self.assertIn("verification_pressure", record.current_em_state)
        self.assertAlmostEqual(
            record.delta_vs_neutral["verification_pressure"],
            record.current_em_state["verification_pressure"] - 0.1,
        )
        self.assertAlmostEqual(
            record.delta_vs_current["verification_pressure"],
            record.current_em_state["verification_pressure"] - 0.3,
        )
        self.assertGreaterEqual(record.total_mass, 0.0)
        self.assertAlmostEqual(record.total_mass, record.positive_mass + record.negative_mass)
        self.assertIn("mock:verification", record.sae_feature_values)
        self.assertIn("mock:verification", record.sae_feature_mask)
        self.assertEqual(record.layer_selection_policy, "mock_single_site_l0_resid_prefill")
        self.assertEqual(record.prompt_template_id, "raw_chunk_v1")
        self.assertEqual(record.normalization_policy, "mock_unit_interval_keyword_scores")

    def test_command_provider_maps_sidecar_selector_rows_to_activation_records(self):
        doc = DocumentRecord.from_text(
            source_uri="memory://real-telemetry",
            title="Telemetry",
            text="Needle evidence lives in this chunk.",
        )
        chunk = Chunker(ChunkerSettings(chunk_size=20, chunk_overlap=0)).split([doc])[0]
        with tempfile.TemporaryDirectory() as tmp:
            script = self._write_fake_capture_script(Path(tmp))
            provider = CommandPrefillTelemetryProvider(
                command=[sys.executable, str(script), "{input_jsonl}", "{output_jsonl}"],
                provider_id="sidecar-command-prefill",
                model_id="gemma4-e4b",
                site_id="l16_resid_pre",
                layer_selection_policy="semantic_middle_late_resid_pre_l16",
                prompt_template_id="rag_raw_chunk_prefill_v1",
                normalization_policy="sidecar_centered_unit_norm_v1",
            )

            record = provider.capture_prefill([chunk])[0]

        self.assertEqual(record.chunk_id, chunk.chunk_id)
        self.assertEqual(record.document_id, chunk.document_id)
        self.assertEqual(record.provider_id, "sidecar-command-prefill")
        self.assertEqual(record.model_id, "gemma4-e4b")
        self.assertEqual(record.site_id, "l16_resid_pre")
        self.assertEqual(record.layer_index, 16)
        self.assertEqual(record.layer_selection_policy, "semantic_middle_late_resid_pre_l16")
        self.assertEqual(record.prompt_template_id, "rag_raw_chunk_prefill_v1")
        self.assertEqual(record.normalization_policy, "sidecar_centered_unit_norm_v1")
        self.assertTrue(record.telemetry_valid)
        self.assertAlmostEqual(record.current_em_state["evidence_pressure"], 0.75)
        self.assertAlmostEqual(record.neutral_baseline_state["evidence_pressure"], 0.25)
        self.assertAlmostEqual(record.delta_vs_neutral["evidence_pressure"], 0.5)
        self.assertAlmostEqual(record.delta_vs_current["evidence_pressure"], 0.35)
        self.assertIn("sae:needle", record.sae_feature_values)
        self.assertAlmostEqual(record.sae_delta_vs_current["sae:needle"], 0.6)
        self.assertEqual(record.provenance["capture_phase"], "prefill")
        self.assertEqual(record.provenance["generation_disabled"], True)
        self.assertEqual(record.provenance["requested_prompt_section_label"], "document_chunk")
        self.assertEqual(record.provenance["prompt_section_label"], "document_chunk")

    def test_command_provider_accepts_prefill_window_extracted_from_decode_tail_request(self):
        doc = DocumentRecord.from_text(
            source_uri="memory://real-telemetry",
            title="Telemetry",
            text="Needle evidence lives in this chunk.",
        )
        chunk = Chunker(ChunkerSettings(chunk_size=20, chunk_overlap=0)).split([doc])[0]
        with tempfile.TemporaryDirectory() as tmp:
            script = self._write_fake_capture_script(Path(tmp), prefill_only_extracted=True)
            provider = CommandPrefillTelemetryProvider(
                command=[sys.executable, str(script), "{input_jsonl}", "{output_jsonl}"],
                provider_id="sidecar-command-prefill",
                model_id="deepseek-r1-distill-llama-8b-q6",
                site_id="emv2_prefill_last",
                layer_selection_policy="emv2_selected_prefill_windows",
                prompt_template_id="rag_chat_user_prefill_v1",
                normalization_policy="prefill_last_summary_chunk8_v1",
            )

            record = provider.capture_prefill([chunk])[0]

        self.assertTrue(record.provenance["prefill_only_extracted"])
        self.assertEqual(record.provenance["source_completion_tokens"], 1)
        self.assertFalse(record.provenance["source_generation_disabled"])
        self.assertEqual(record.provenance["generation_disabled"], False)
        self.assertEqual(record.provenance["capture_phase"], "prefill")

    def test_command_provider_reuses_cached_prefill_records(self):
        doc = DocumentRecord.from_text(
            source_uri="memory://cached-telemetry",
            title="Telemetry",
            text="Needle evidence lives in this cached chunk.",
        )
        chunk = Chunker(ChunkerSettings(chunk_size=20, chunk_overlap=0)).split([doc])[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = self._write_fake_capture_script(root)
            counter = root / "capture-count.txt"
            provider = CommandPrefillTelemetryProvider(
                command=[sys.executable, str(script), "{input_jsonl}", "{output_jsonl}", str(counter)],
                provider_id="sidecar-command-prefill",
                model_id="gemma4-e4b",
                site_id="l16_resid_pre",
                layer_selection_policy="semantic_middle_late_resid_pre_l16",
                prompt_template_id="rag_raw_chunk_prefill_v1",
                normalization_policy="sidecar_centered_unit_norm_v1",
                cache_dir=root / "cache",
            )

            first = provider.capture_prefill([chunk])[0]
            second = provider.capture_prefill([chunk])[0]
            capture_count = counter.read_text(encoding="utf-8").strip()

        self.assertEqual(first.chunk_id, second.chunk_id)
        self.assertEqual(first.sae_feature_values, second.sae_feature_values)
        self.assertEqual(capture_count, "1")

    def test_command_provider_ignores_invalid_cache_rows_and_recaptures(self):
        doc = DocumentRecord.from_text(
            source_uri="memory://invalid-cached-telemetry",
            title="Telemetry",
            text="Needle evidence lives in this cached chunk.",
        )
        chunk = Chunker(ChunkerSettings(chunk_size=20, chunk_overlap=0)).split([doc])[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = self._write_fake_capture_script(root)
            counter = root / "capture-count.txt"
            provider = CommandPrefillTelemetryProvider(
                command=[sys.executable, str(script), "{input_jsonl}", "{output_jsonl}", str(counter)],
                provider_id="sidecar-command-prefill",
                model_id="gemma4-e4b",
                site_id="l16_resid_pre",
                layer_selection_policy="semantic_middle_late_resid_pre_l16",
                prompt_template_id="rag_raw_chunk_prefill_v1",
                normalization_policy="sidecar_centered_unit_norm_v1",
                cache_dir=root / "cache",
            )
            cache_path = provider._cache_path(chunk)
            assert cache_path is not None
            cache_path.parent.mkdir(parents=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "capture_run_id": "invalid",
                        "telemetry_valid": False,
                        "invalid_reason": "raw activation manifest not found",
                        "sae_feature_values": {},
                    }
                ),
                encoding="utf-8",
            )

            record = provider.capture_prefill([chunk])[0]
            capture_count = counter.read_text(encoding="utf-8").strip()

        self.assertTrue(record.telemetry_valid)
        self.assertEqual(capture_count, "1")
        self.assertIn("sae:needle", record.sae_feature_values)

    def test_command_provider_rejects_invalid_capture_rows(self):
        doc = DocumentRecord.from_text(
            source_uri="memory://invalid-telemetry",
            title="Telemetry",
            text="Needle evidence lives in this chunk.",
        )
        chunk = Chunker(ChunkerSettings(chunk_size=20, chunk_overlap=0)).split([doc])[0]
        with tempfile.TemporaryDirectory() as tmp:
            script = self._write_invalid_capture_script(Path(tmp))
            provider = CommandPrefillTelemetryProvider(
                command=[sys.executable, str(script), "{input_jsonl}", "{output_jsonl}"],
                provider_id="sidecar-command-prefill",
                model_id="gemma4-e4b",
                site_id="l16_resid_pre",
                layer_selection_policy="semantic_middle_late_resid_pre_l16",
                prompt_template_id="rag_raw_chunk_prefill_v1",
                normalization_policy="sidecar_centered_unit_norm_v1",
                cache_dir=Path(tmp) / "cache",
            )

            with self.assertRaisesRegex(ValueError, "telemetry row .* is invalid"):
                provider.capture_prefill([chunk])

    def test_command_provider_supports_query_prefill_activation_search(self):
        docs = [
            DocumentRecord.from_text("memory://needle", "Needle", "Needle evidence belongs here."),
            DocumentRecord.from_text("memory://distractor", "Distractor", "Travel policy belongs elsewhere."),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            script = self._write_fake_capture_script(Path(tmp))
            provider = CommandPrefillTelemetryProvider(
                command=[sys.executable, str(script), "{input_jsonl}", "{output_jsonl}"],
                provider_id="sidecar-command-prefill",
                model_id="gemma4-e4b",
                site_id="l16_resid_pre",
                layer_selection_policy="semantic_middle_late_resid_pre_l16",
                prompt_template_id="rag_raw_chunk_prefill_v1",
                normalization_policy="sidecar_centered_unit_norm_v1",
            )
            engine = RagEngine(
                chunker=Chunker(ChunkerSettings(chunk_size=20, chunk_overlap=0)),
                embedder=HashEmbeddingProvider(dimension=32),
                telemetry_provider=provider,
            )
            chunks = engine.ingest(docs)

            results = engine.search_activation_knn("needle evidence", top_k=1)

        self.assertEqual(results[0].chunk_id, chunks[0].chunk_id)
        self.assertEqual(results[0].strategy, "activation-sim")

    def _write_fake_capture_script(self, root: Path, *, prefill_only_extracted: bool = False) -> Path:
        script = root / "fake_prefill_capture.py"
        generation_disabled = "False" if prefill_only_extracted else 'row["generation_disabled"]'
        extra_provenance = (
            '"prefill_only_extracted": True, "source_completion_tokens": 1, "source_generation_disabled": False,'
            if prefill_only_extracted
            else ""
        )
        script_text = """
import json
import sys
from pathlib import Path

input_path, output_path = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3:
    counter = Path(sys.argv[3])
    count = int(counter.read_text(encoding="utf-8")) if counter.exists() else 0
    counter.write_text(str(count + 1), encoding="utf-8")
with open(input_path, "r", encoding="utf-8") as source, open(output_path, "w", encoding="utf-8") as sink:
    for line in source:
        row = json.loads(line)
        text = row["text"].lower()
        active = 1.0 if "needle" in text or "evidence" in text else 0.0
        current = {"evidence_pressure": 0.75 * active, "travel_pressure": 0.75 * (1.0 - active)}
        baseline = {"evidence_pressure": 0.25, "travel_pressure": 0.25}
        prior = {"evidence_pressure": 0.40, "travel_pressure": 0.40}
        out = {
            "schema_version": "vicuna.rmt_span_selector.row.v1",
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "capture_run_id": row["capture_run_id"],
            "provider_id": "sidecar-command-prefill",
            "model_id": "gemma4-e4b",
            "site_id": "l16_resid_pre",
            "hook_name": "model.layers.16.resid_pre",
            "layer_index": 16,
            "layer_selection_policy": row["layer_selection_policy"],
            "prompt_template_id": row["prompt_template_id"],
            "prompt_template_hash": row["prompt_template_hash"],
            "prompt_section_label": row["requested_prompt_section_label"],
            "requested_prompt_section_label": row["requested_prompt_section_label"],
            "raw_prompt_section_label": "prompt",
            "normalization_policy": row["normalization_policy"],
            "token_start": 0,
            "token_end": row["token_count_estimate"],
            "current_em_state": current,
            "neutral_baseline_state": baseline,
            "prior_current_state": prior,
            "em_delta_vs_neutral": {key: current[key] - baseline[key] for key in current},
            "span_delta_vs_current": {key: current[key] - prior[key] for key in current},
            "saturation": {key: abs(current[key]) for key in current},
            "residual_headroom": {key: 1.0 - abs(current[key]) for key in current},
            "sae_feature_values": {"sae:needle": active, "sae:travel": 1.0 - active},
            "sae_delta_vs_neutral": {"sae:needle": active - 0.2, "sae:travel": (1.0 - active) - 0.2},
            "sae_delta_vs_current": {"sae:needle": active - 0.4, "sae:travel": (1.0 - active) - 0.4},
            "sae_feature_mask": {"sae:needle": bool(active), "sae:travel": not bool(active)},
            "telemetry_valid": True,
            "provenance": {
                "capture_phase": row["capture_phase"],
                "generation_disabled": __GENERATION_DISABLED__,
                __EXTRA_PROVENANCE__
            },
        }
        sink.write(json.dumps(out) + "\\n")
""".lstrip()
        script_text = script_text.replace("__GENERATION_DISABLED__", generation_disabled)
        script_text = script_text.replace("__EXTRA_PROVENANCE__", extra_provenance)
        script.write_text(script_text, encoding="utf-8")
        return script

    def _write_invalid_capture_script(self, root: Path) -> Path:
        script = root / "invalid_prefill_capture.py"
        script.write_text(
            """
import json
import sys

input_path, output_path = sys.argv[1], sys.argv[2]
with open(input_path, "r", encoding="utf-8") as source, open(output_path, "w", encoding="utf-8") as sink:
    for line in source:
        row = json.loads(line)
        sink.write(json.dumps({
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "capture_run_id": row["capture_run_id"],
            "provider_id": row["provider_id"],
            "model_id": row["model_id"],
            "site_id": row["site_id"],
            "layer_selection_policy": row["layer_selection_policy"],
            "prompt_template_id": row["prompt_template_id"],
            "prompt_template_hash": row["prompt_template_hash"],
            "normalization_policy": row["normalization_policy"],
            "capture_phase": "prefill",
            "generation_disabled": True,
            "prefill_only_extracted": False,
            "sae_feature_values": {},
            "telemetry_valid": False,
            "invalid_reason": "raw activation manifest not found",
            "provenance": {"capture_phase": "prefill", "generation_disabled": True},
        }) + "\\n")
""".lstrip(),
            encoding="utf-8",
        )
        return script


if __name__ == "__main__":
    unittest.main()
