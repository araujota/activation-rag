import importlib.util
import unittest
from pathlib import Path


def load_capture_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "capture_sidecar_prefill.py"
    spec = importlib.util.spec_from_file_location("capture_sidecar_prefill", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SidecarCaptureScriptTests(unittest.TestCase):
    def test_prefill_feature_subset_keeps_prefill_windows_only(self):
        module = load_capture_module()

        subset = module.prefill_feature_subset(
            {
                "act:site_a:prefill_last:mean": 0.1,
                "act:site_a:post50_mean:mean": 0.2,
                "act:site_a:decode_first:mean": 0.3,
                "act:site_a:traj_prefill_mean:mean": 0.4,
                "act:site_a:traj_decode_mean:mean": 0.5,
            }
        )

        self.assertEqual(
            subset,
            {
                "act:site_a:prefill_last:mean": 0.1,
                "act:site_a:post50_mean:mean": 0.2,
                "act:site_a:traj_prefill_mean:mean": 0.4,
            },
        )

    def test_build_output_row_preserves_section_and_prefill_filter_provenance(self):
        module = load_capture_module()
        request = {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "capture_run_id": "run-1",
            "provider_id": "sidecar",
            "model_id": "model",
            "site_id": "site",
            "layer_selection_policy": "layers",
            "prompt_template_id": "template",
            "prompt_template_hash": "hash",
            "normalization_policy": "norm",
            "token_count_estimate": 7,
            "requested_prompt_section_label": "document_chunk",
        }

        row = module.build_output_row(
            request,
            features={"act:site:prefill_last:mean": 0.5},
            request_id="req-1",
            activation_dir="/tmp/req-1",
            response_usage={"prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4},
            endpoint="/v1/chat/completions",
            manifest_stats={"raw_manifest_rows": 18, "accepted_prompt_prefill_rows": 18},
        )

        self.assertEqual(row["capture_phase"], "prefill")
        self.assertTrue(row["generation_disabled"])
        self.assertTrue(row["provenance"]["prefill_only_extracted"])
        self.assertEqual(row["provenance"]["source_completion_tokens"], 0)
        self.assertEqual(row["prompt_section_label"], "document_chunk")
        self.assertEqual(row["requested_prompt_section_label"], "document_chunk")
        self.assertEqual(row["raw_prompt_section_label"], "prompt")
        self.assertEqual(row["provenance"]["manifest_filter"]["accepted_prompt_prefill_rows"], 18)
        self.assertEqual(row["sae_feature_values"], {"act:site:prefill_last:mean": 0.5})

    def test_build_output_row_marks_strict_zero_token_capture(self):
        module = load_capture_module()
        request = {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "capture_run_id": "run-1",
            "provider_id": "sidecar",
            "model_id": "model",
            "site_id": "site",
            "layer_selection_policy": "layers",
            "prompt_template_id": "template",
            "prompt_template_hash": "hash",
            "normalization_policy": "norm",
            "token_count_estimate": 7,
        }

        row = module.build_output_row(
            request,
            features={"act:site:prefill_last:mean": 0.5},
            request_id="req-1",
            activation_dir="/tmp/req-1",
            response_usage={"prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4},
            endpoint="/v1/chat/completions",
            raw_activation_reused=True,
        )

        self.assertTrue(row["generation_disabled"])
        self.assertTrue(row["provenance"]["prefill_only_extracted"])
        self.assertEqual(row["provenance"]["source_completion_tokens"], 0)
        self.assertTrue(row["provenance"]["raw_activation_reused"])
        self.assertEqual(row["prompt_section_label"], "document_chunk")


if __name__ == "__main__":
    unittest.main()
