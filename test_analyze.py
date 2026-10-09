import unittest

from analyze import length_curve, paired_comparison, training_diagnostics


def row(prompt, correct, tokens=20, thinking=10, truncated=False):
    return {"prompt": prompt, "target": 1, "correct": correct,
            "total_tokens": tokens, "thinking_tokens": thinking,
            "finished": not truncated, "truncated": truncated}


class AnalyzeTests(unittest.TestCase):
    def test_paired_bootstrap_is_reproducible_and_preserves_pairs(self):
        base = [row(str(i), False) for i in range(8)]
        arm = [row(str(i), True, 10, 5) for i in range(8)]
        result = paired_comparison(base, arm, 30, 7)
        self.assertEqual(result, paired_comparison(base, arm, 30, 7))
        self.assertEqual(result["delta_arm_minus_reference"]["accuracy"], 1)
        self.assertEqual(result["delta_95pct_paired_bootstrap_ci"]["accuracy"], [1, 1])
        self.assertEqual(result["arm"]["correct_per_1000_tokens"], 100)
        with self.assertRaisesRegex(ValueError, "cannot pair"):
            paired_comparison(base, list(reversed(arm)), 5)

    def test_empty_correct_subset_and_length_bins(self):
        records = [row("a", False, 30, 100, True), row("b", False, 20, 500)]
        result = paired_comparison(records, records, 10)
        self.assertIsNone(result["delta_arm_minus_reference"]["median_thinking_correct"])
        self.assertEqual(result["delta_95pct_paired_bootstrap_ci"]["median_thinking_correct"], [None, None])
        curve = length_curve(records, [256, 512])
        self.assertEqual([x["n"] for x in curve], [1, 1, 0])
        self.assertEqual(curve[0]["truncated_rate"], 1)
        self.assertIsNone(curve[-1]["accuracy"])

    def test_training_log_blocks(self):
        records = [{"correct": i == 0, "truncated": i == 1,
                    "finished_thinking": i != 1, "thinking_tokens": i + 1,
                    "completion_tokens": 5, "penalty": -.25 if i == 0 else 0,
                    "reward": 1.75 if i == 0 else 0} for i in range(3)]
        result = training_diagnostics(records, 2)
        self.assertEqual(result["overall"]["correct_rate"], 1 / 3)
        self.assertEqual([b["n"] for b in result["blocks"]], [2, 1])
        self.assertEqual(result["overall"]["mean_penalty"], -.25 / 3)


if __name__ == "__main__":
    unittest.main()
