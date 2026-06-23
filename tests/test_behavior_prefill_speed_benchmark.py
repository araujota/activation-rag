import importlib.util
import unittest
from pathlib import Path


def load_speed_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_behavior_prefill_speed.py"
    spec = importlib.util.spec_from_file_location("benchmark_behavior_prefill_speed", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BehaviorPrefillSpeedBenchmarkTests(unittest.TestCase):
    def test_build_capture_requests_uses_query_candidate_prompt(self):
        module = load_speed_module()
        groups = [
            {
                "query_id": "q1",
                "query_text": "What does the API return?",
                "candidates": [
                    {
                        "chunk_id": "c1",
                        "text": "The API returns JSON.",
                        "label": 1,
                    }
                ],
            }
        ]

        requests = module.build_capture_requests(groups, dataset_name="fixture")

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["prompt_template_id"], "behavior_support_pair_v1")
        self.assertEqual(requests[0]["requested_prompt_section_label"], "query_candidate_behavior_prompt")
        self.assertIn("Query:\nWhat does the API return?", requests[0]["prompt_text"])
        self.assertIn("Candidate evidence:\nThe API returns JSON.", requests[0]["prompt_text"])
        self.assertIn("Answer support:", requests[0]["prompt_text"])

    def test_trim_groups_preserves_candidate_order(self):
        module = load_speed_module()
        groups = [
            {"query_id": "q1", "candidates": [{"chunk_id": "c1"}, {"chunk_id": "c2"}]},
            {"query_id": "q2", "candidates": [{"chunk_id": "c3"}]},
        ]

        trimmed = module.trim_groups(groups, query_limit=1, candidates_per_query=1)

        self.assertEqual(len(trimmed), 1)
        self.assertEqual([candidate["chunk_id"] for candidate in trimmed[0]["candidates"]], ["c1"])


if __name__ == "__main__":
    unittest.main()
