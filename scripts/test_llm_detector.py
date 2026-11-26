
import sys
import json
import unittest

from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.defenses.llm_detector import LLMDetector


class MockClient:

    def __init__(self, label="BENIGN", fail=False):

        self.label = label

        self.fail = fail

        self.calls = 0

        self.chat = SimpleNamespace(

            completions=SimpleNamespace(

                create=self.create
            )
        )

    def create(self, **kwargs):

        self.calls += 1

        if self.fail:

            raise RuntimeError(
                "Simulated API failure"
            )

        content = json.dumps({

            "label": self.label,

            "category": "test_category",

            "reason": "Mock classification"
        })

        return SimpleNamespace(

            choices=[

                SimpleNamespace(

                    message=SimpleNamespace(

                        content=content
                    )
                )
            ],

            usage=SimpleNamespace(

                prompt_tokens=100,

                completion_tokens=20,

                total_tokens=120
            )
        )


class TestLLMDetector(unittest.TestCase):

    def test_benign_classification(self):

        detector = LLMDetector(

            client=MockClient("BENIGN")
        )

        result = detector.scan(
            "What is the library policy?"
        )

        self.assertFalse(result["detected"])

        self.assertEqual(
            result["action"], "allow"
        )

    def test_adversarial_classification(self):

        detector = LLMDetector(

            client=MockClient("ADVERSARIAL")
        )

        result = detector.scan(
            "Synthetic security test fixture."
        )

        self.assertTrue(result["detected"])

        self.assertEqual(
            result["action"], "block"
        )

    def test_invalid_json(self):

        with self.assertRaises(ValueError):

            LLMDetector.parse_response(
                "invalid response"
            )

    def test_invalid_label(self):

        with self.assertRaises(ValueError):

            LLMDetector.parse_response(

                json.dumps({

                    "label": "INVALID",

                    "category": "test",

                    "reason": "test"
                })
            )

    def test_api_failure(self):

        detector = LLMDetector(

            client=MockClient(fail=True)
        )

        detector.failure_policy = "block"

        result = detector.scan(
            "Library question"
        )

        self.assertIsNone(result["detected"])

        self.assertEqual(
            result["label"], "UNKNOWN"
        )

        self.assertEqual(
            result["action"], "block"
        )

    def test_token_accounting(self):

        detector = LLMDetector(

            client=MockClient()
        )

        result = detector.scan(
            "Library question"
        )

        self.assertEqual(
            result["total_tokens"], 120
        )

    def test_context_filtering(self):

        detector = LLMDetector(

            client=MockClient("ADVERSARIAL")
        )

        chunks = [

            {
                "chunk_id": "TEST_CHUNK_1",

                "document_id": "TEST_DOC_1",

                "text":
                    "Synthetic test content."
            }
        ]

        result = detector.scan_context(
            chunks
        )

        self.assertEqual(
            result["flagged_chunk_count"], 1
        )

        self.assertEqual(
            result["allowed_chunk_count"], 0
        )

    def test_invalid_input(self):

        detector = LLMDetector(

            client=MockClient()
        )

        with self.assertRaises(TypeError):

            detector.scan(None)


if __name__ == "__main__":

    unittest.main(verbosity=2)
