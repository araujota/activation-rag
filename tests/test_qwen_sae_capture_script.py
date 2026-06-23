import importlib.util
import json
import sys
import unittest
from pathlib import Path


def load_capture_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "capture_qwen_sae_prefill.py"
    spec = importlib.util.spec_from_file_location("capture_qwen_sae_prefill", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QwenSaeCaptureScriptTests(unittest.TestCase):
    def test_build_episode_rows_preserves_chunk_identity_and_prompt_text(self):
        module = load_capture_module()
        requests = [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "text": "raw chunk text",
                "prompt_text": "canonical prompt text",
                "text_hash": "hash-1",
                "capture_run_id": "run-1",
            }
        ]

        episodes = module.build_episode_rows(requests)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["episode_id"], "activation-rag-prefill-chunk-1")
        self.assertEqual(episodes[0]["candidate_spans"][0]["span_id"], "chunk-1")
        self.assertEqual(episodes[0]["candidate_spans"][0]["text_preview"], "canonical prompt text")
        self.assertEqual(episodes[0]["candidate_spans"][0]["text_hash"], "hash-1")

    def test_adapt_sae_row_emits_selector_compatible_prefill_telemetry(self):
        module = load_capture_module()
        request = {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "capture_run_id": "run-1",
            "provider_id": "qwen-sae",
            "model_id": "qwen3-rmt",
            "site_id": "l07_resid_pre",
            "layer_selection_policy": "l07_resid_pre_core245",
            "prompt_template_id": "template-v1",
            "prompt_template_hash": "hash",
            "normalization_policy": "sae_checkpoint_normalization",
            "token_count_estimate": 12,
            "requested_prompt_section_label": "document_chunk",
        }
        sae_row = {
            "span_id": "chunk-1",
            "evidence_mode": "qwen_no_memory_span_preview",
            "layer_index": 7,
            "qwen_seq_tokens": 128,
            "sequence_tokens": 18,
            "sae_feature_values": {"5439": 1.25, "18172": 0.5},
            "max_token_offset_by_feature": {"5439": 4, "18172": 9},
        }

        row = module.adapt_sae_row(
            request=request,
            sae_row=sae_row,
            feature_manifest_summary={"feature_set_id": "core245", "feature_count": 245},
            encoder_summary={"evidence_mode": "qwen_no_memory_span_preview"},
        )

        self.assertEqual(row["schema_version"], "activation_rag.qwen_sae_prefill_capture.row.v1")
        self.assertEqual(row["chunk_id"], "chunk-1")
        self.assertEqual(row["sae_feature_values"], {"5439": 1.25, "18172": 0.5})
        self.assertEqual(row["sae_delta_vs_neutral"], {"5439": 1.25, "18172": 0.5})
        self.assertEqual(row["aggregation"], "selected_sae_max_over_prompt_tokens")
        self.assertTrue(row["generation_disabled"])
        self.assertTrue(row["telemetry_valid"])
        self.assertEqual(row["provenance"]["feature_set_id"], "core245")
        self.assertEqual(row["provenance"]["max_token_offset_by_feature"], {"5439": 4, "18172": 9})

    def test_split_timing_summary_removes_machine_readable_line(self):
        module = load_capture_module()
        stderr, summary = module.split_timing_summary(
            'warning line\nACTIVATION_RAG_QWEN_SAE_TIMING {"duration_s": 1.5, "row_count": 4}\n'
        )

        self.assertEqual(stderr, "warning line\n")
        self.assertEqual(summary, {"duration_s": 1.5, "row_count": 4})

    def test_remote_code_contains_opt_in_optimized_modes(self):
        module = load_capture_module()

        self.assertIn("early_stop_layer", module.REMOTE_CAPTURE_CODE)
        self.assertIn("_activation_rag_stop_after_capture", module.REMOTE_CAPTURE_CODE)
        self.assertIn("early_stop_layer_prefix_cache", module.REMOTE_CAPTURE_CODE)
        self.assertIn("DynamicCache", module.REMOTE_CAPTURE_CODE)
        self.assertIn("optimized_batch_size", module.REMOTE_CAPTURE_CODE)
        self.assertIn("summarize_hidden_batch", module.REMOTE_CAPTURE_CODE)

    def test_run_remote_capture_supports_local_host(self):
        module = load_capture_module()
        original = module.REMOTE_CAPTURE_CODE
        module.REMOTE_CAPTURE_CODE = (
            "import json, os, pathlib; "
            "payload = json.loads(pathlib.Path(os.environ['ACTIVATION_RAG_QWEN_SAE_PAYLOAD']).read_text()); "
            "print(json.dumps(payload, sort_keys=True))"
        )
        try:
            completed = module.run_remote_capture("local", sys.executable, {"ok": True})
        finally:
            module.REMOTE_CAPTURE_CODE = original

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(json.loads(completed.stdout), {"ok": True})


if __name__ == "__main__":
    unittest.main()
