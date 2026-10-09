import math
import unittest

from qwen_normal import final_answer_ids, length_penalty, thinking_span


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(token) for token in ids)


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharTokenizer()

    def answer(self, last_line):
        ids = self.tokenizer.encode("thinking</think>\n" + last_line)
        return final_answer_ids(ids, self.tokenizer, "qwen", "<think>")

    def test_final_answer_needs_one_integer_on_the_last_line(self):
        self.assertEqual(self.answer("#### 42"), 42)
        self.assertEqual(self.answer("#### -1,234"), -1234)
        self.assertEqual(self.answer("#### <42>"), 42)
        self.assertEqual(self.answer("#### <integer>42"), 42)
        for text in ("#### 42.5", "#### 42abc", "#### 1,23", "#### 42\nextra", "#### <integer>"):
            with self.subTest(text=text):
                self.assertIsNone(self.answer(text))
        self.assertIsNone(final_answer_ids(self.tokenizer.encode("#### 42"), self.tokenizer, "qwen", "<think>"))

    def test_penalties_match_at_the_anchor(self):
        norm = 1150
        for strength in (0.125, 0.5):
            ds = length_penalty(norm, "deepseek", norm, strength, strength, 1, 1 / norm, 1.5)
            composer = length_penalty(norm, "composer", norm, strength, strength, 1, 1 / norm, 1.5)
            self.assertTrue(math.isclose(ds, -strength))
            self.assertTrue(math.isclose(composer, ds))


if __name__ == "__main__":
    unittest.main()
