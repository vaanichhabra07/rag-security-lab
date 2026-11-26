
import sys
import unittest

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from test_defense_pipeline import (
    MockRetriever,
    MockLLM
)

from test_llm_detector import MockClient

from src.defenses.llm_detector import LLMDetector

from src.rag.rag_pipeline import RAGPipeline


class TestD2Pipeline(unittest.TestCase):

    def build_pipeline(
        self,
        label="BENIGN",
        fail=False
    ):

        retriever = MockRetriever()

        llm = MockLLM()

        client = MockClient(
            label=label,
            fail=fail
        )

        detector = LLMDetector(
            client=client
        )

        detector.failure_policy = "block"

        pipeline = RAGPipeline(

            retriever=retriever,

            llm_client=llm,

            llm_detector=detector
        )

        return pipeline, retriever, llm, client

    def test_d2_benign_request(self):

        pipeline, _, llm, client = (
            self.build_pipeline()
        )

        result = pipeline.ask(

            "What is the library policy?",

            defense_mode="D2"
        )

        self.assertFalse(result["blocked"])

        self.assertTrue(result["llm_called"])

        self.assertEqual(llm.calls, 1)

        self.assertEqual(client.calls, 2)

        self.assertEqual(
            result["defense_api_calls"], 2
        )

    def test_d2_adversarial_request(self):

        pipeline, retriever, llm, _ = (
            self.build_pipeline(
                label="ADVERSARIAL"
            )
        )

        result = pipeline.ask(

            "Synthetic security test fixture.",

            defense_mode="D2"
        )

        self.assertTrue(result["blocked"])

        self.assertTrue(result["attack_detected"])

        self.assertEqual(llm.calls, 0)

        self.assertEqual(retriever.calls, 0)

    def test_d2_api_failure(self):

        pipeline, _, llm, _ = (
            self.build_pipeline(fail=True)
        )

        result = pipeline.ask(

            "Library question",

            defense_mode="D2"
        )

        self.assertTrue(result["blocked"])

        self.assertFalse(result["attack_detected"])

        self.assertEqual(
            result["classifier_errors"], 1
        )

        self.assertEqual(
            result["execution_status"],
            "defense_error"
        )

        self.assertEqual(llm.calls, 0)

    def test_d2_token_accounting(self):

        pipeline, _, _, _ = (
            self.build_pipeline()
        )

        result = pipeline.ask(

            "Library question",

            defense_mode="D2"
        )

        self.assertEqual(
            result["defense_total_tokens"], 240
        )

        self.assertEqual(
            result["combined_total_tokens"], 256
        )

    def test_d0_does_not_call_d2(self):

        pipeline, _, _, client = (
            self.build_pipeline()
        )

        pipeline.ask(

            "Library question",

            defense_mode="D0"
        )

        self.assertEqual(client.calls, 0)

    def test_d1_does_not_call_d2(self):

        pipeline, _, _, client = (
            self.build_pipeline()
        )

        pipeline.ask(

            "Library question",

            defense_mode="D1"
        )

        self.assertEqual(client.calls, 0)


if __name__ == "__main__":

    unittest.main(verbosity=2)
