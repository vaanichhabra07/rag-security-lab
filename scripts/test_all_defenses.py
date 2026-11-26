
import sys
import json
import unittest

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.rag.rag_pipeline import RAGPipeline

from src.defenses.context_isolation import (
    ContextIsolationDefense
)

from src.defenses.output_validator import (
    OutputValidator
)

from src.defenses.combined_defense import (
    CombinedDefense
)

from src.defenses.llm_detector import LLMDetector

from test_llm_detector import MockClient


TEST_QUESTION = "What is the library borrowing policy?"

TEST_CHUNKS = [
    {
        "rank": 1,
        "chunk_id": "TEST_CHUNK_001",
        "document_id": "TEST_DOC_001",
        "domain": "library",
        "source": "synthetic_test.txt",
        "text": (
            "Students may borrow five library books."
        ),
        "distance": 0.1,
        "similarity": 0.9
    }
]


class MockRetriever:

    def __init__(self):

        self.calls = 0

    def retrieve(self, query, top_k=3, domain=None):

        self.calls += 1

        return {
            "retrieved_chunks": [
                dict(chunk) for chunk in TEST_CHUNKS
            ],
            "retrieval_latency_ms": 2.0
        }


class MockLLM:

    provider = "mock"

    model = "mock-model"

    def __init__(
        self,
        answer="Students may borrow five books."
    ):

        self.answer = answer

        self.calls = 0

        self.last_system_prompt = None

        self.last_user_prompt = None

    def generate(self, system_prompt, user_prompt):

        self.calls += 1

        self.last_system_prompt = system_prompt

        self.last_user_prompt = user_prompt

        return {
            "answer": self.answer,
            "provider": self.provider,
            "model": self.model,
            "generation_latency_ms": 5.0,
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
            "finish_reason": "stop"
        }


class TestContextIsolation(unittest.TestCase):

    def test_structured_context(self):

        defense = ContextIsolationDefense()

        result = defense.build_prompt(
            TEST_QUESTION,
            TEST_CHUNKS
        )

        payload = json.loads(result["user_prompt"])

        self.assertEqual(
            payload["student_question"],
            TEST_QUESTION
        )

        self.assertEqual(
            len(payload["retrieved_documents"]),
            1
        )

    def test_trusted_prompt_separation(self):

        defense = ContextIsolationDefense()

        result = defense.build_prompt(
            TEST_QUESTION,
            TEST_CHUNKS
        )

        self.assertNotIn(
            TEST_CHUNKS[0]["text"],
            result["system_prompt"]
        )

    def test_context_preserved(self):

        defense = ContextIsolationDefense()

        result = defense.build_prompt(
            TEST_QUESTION,
            TEST_CHUNKS
        )

        payload = json.loads(result["user_prompt"])

        self.assertEqual(
            payload["retrieved_documents"][0]["content"],
            TEST_CHUNKS[0]["text"]
        )

    def test_empty_question(self):

        with self.assertRaises(ValueError):

            ContextIsolationDefense().build_prompt(
                "",
                TEST_CHUNKS
            )


class TestOutputValidator(unittest.TestCase):

    def test_normal_answer(self):

        result = OutputValidator().validate(
            "Students may borrow five books."
        )

        self.assertTrue(result["valid"])

        self.assertFalse(result["blocked"])

    def test_protected_value(self):

        validator = OutputValidator(
            protected_values=[
                "SYNTHETIC_PRIVATE_TEST_VALUE"
            ]
        )

        result = validator.validate(
            "SYNTHETIC_PRIVATE_TEST_VALUE"
        )

        self.assertTrue(result["blocked"])

        self.assertNotIn(
            "SYNTHETIC_PRIVATE_TEST_VALUE",
            result["answer"]
        )

    def test_system_prompt_disclosure(self):

        prompt = (
            "These are trusted application instructions "
            "used exclusively for the institutional "
            "assistant and its internal configuration."
        )

        result = OutputValidator().validate(
            answer=prompt,
            system_prompt=prompt
        )

        self.assertTrue(result["blocked"])

    def test_security_discussion_allowed(self):

        result = OutputValidator().validate(
            "Prompt injection is an AI security risk."
        )

        self.assertFalse(result["blocked"])

    def test_empty_response(self):

        result = OutputValidator().validate("")

        self.assertFalse(result["valid"])

        self.assertEqual(
            result["validation_status"],
            "empty_response"
        )


class TestDefenseConfiguration(unittest.TestCase):

    def test_d0(self):

        config = CombinedDefense.get_configuration(
            "D0"
        )

        self.assertEqual(config.active_layers(), [])

    def test_d5(self):

        config = CombinedDefense.get_configuration(
            "D5"
        )

        self.assertEqual(
            config.active_layers(),
            ["D1", "D2", "D3", "D4"]
        )

    def test_pairwise_ablation(self):

        config = CombinedDefense.get_configuration(
            "D1+D3"
        )

        self.assertEqual(
            config.active_layers(),
            ["D1", "D3"]
        )

    def test_all_configurations(self):

        self.assertEqual(
            len(CombinedDefense.supported_modes()),
            12
        )

    def test_invalid_mode(self):

        with self.assertRaises(ValueError):

            CombinedDefense.get_configuration("D9")


class TestCompletePipeline(unittest.TestCase):

    def build_pipeline(
        self,
        answer="Students may borrow five books.",
        label="BENIGN",
        fail=False
    ):

        llm = MockLLM(answer)

        retriever = MockRetriever()

        client = MockClient(
            label=label,
            fail=fail
        )

        detector = LLMDetector(client=client)

        detector.failure_policy = "block"

        validator = OutputValidator(
            protected_values=[
                "SYNTHETIC_PRIVATE_TEST_VALUE"
            ]
        )

        pipeline = RAGPipeline(
            retriever=retriever,
            llm_client=llm,
            llm_detector=detector,
            output_validator=validator
        )

        return pipeline, llm, client

    def test_d0(self):

        pipeline, _, client = self.build_pipeline()

        result = pipeline.ask(
            TEST_QUESTION,
            defense_mode="D0"
        )

        self.assertEqual(
            result["executed_layers"], []
        )

        self.assertEqual(client.calls, 0)

        self.assertTrue(result["llm_called"])

    def test_d1(self):

        pipeline, _, client = self.build_pipeline()

        result = pipeline.ask(
            TEST_QUESTION,
            defense_mode="D1"
        )

        self.assertEqual(
            result["executed_layers"], ["D1"]
        )

        self.assertEqual(client.calls, 0)

    def test_d2(self):

        pipeline, _, client = self.build_pipeline()

        result = pipeline.ask(
            TEST_QUESTION,
            defense_mode="D2"
        )

        self.assertEqual(
            result["executed_layers"], ["D2"]
        )

        self.assertEqual(client.calls, 2)

    def test_d3(self):

        pipeline, llm, client = self.build_pipeline()

        result = pipeline.ask(
            TEST_QUESTION,
            defense_mode="D3"
        )

        self.assertTrue(result["isolation_applied"])

        self.assertEqual(client.calls, 0)

        payload = json.loads(llm.last_user_prompt)

        self.assertIn(
            "retrieved_documents",
            payload
        )

    def test_d4(self):

        pipeline, _, client = self.build_pipeline()

        result = pipeline.ask(
            TEST_QUESTION,
            defense_mode="D4"
        )

        self.assertTrue(
            result["output_validation_applied"]
        )

        self.assertEqual(
            result["output_validation_status"],
            "passed"
        )

        self.assertEqual(client.calls, 0)

    def test_d4_blocks_output(self):

        pipeline, _, _ = self.build_pipeline(
            answer="SYNTHETIC_PRIVATE_TEST_VALUE"
        )

        result = pipeline.ask(
            TEST_QUESTION,
            defense_mode="D4"
        )

        self.assertTrue(result["blocked"])

        self.assertTrue(result["llm_called"])

        self.assertEqual(
            result["blocked_by"], "D4"
        )

        self.assertEqual(
            result["blocked_stage"],
            "generated_output"
        )

        self.assertNotIn(
            "SYNTHETIC_PRIVATE_TEST_VALUE",
            result["answer"]
        )

    def test_d5_all_layers(self):

        pipeline, _, client = self.build_pipeline()

        result = pipeline.ask(
            TEST_QUESTION,
            defense_mode="D5"
        )

        self.assertEqual(
            result["executed_layers"],
            ["D1", "D2", "D3", "D4"]
        )

        self.assertTrue(result["isolation_applied"])

        self.assertTrue(
            result["output_validation_applied"]
        )

        self.assertEqual(client.calls, 2)

    def test_d5_early_termination(self):

        pipeline, llm, _ = self.build_pipeline(
            label="ADVERSARIAL"
        )

        result = pipeline.ask(
            TEST_QUESTION,
            defense_mode="D5"
        )

        self.assertTrue(result["blocked"])

        self.assertEqual(result["blocked_by"], "D2")

        self.assertEqual(llm.calls, 0)

        self.assertFalse(result["isolation_applied"])

        self.assertFalse(
            result["output_validation_applied"]
        )

    def test_d5_classifier_failure(self):

        pipeline, llm, _ = self.build_pipeline(
            fail=True
        )

        result = pipeline.ask(
            TEST_QUESTION,
            defense_mode="D5"
        )

        self.assertTrue(result["blocked"])

        self.assertFalse(result["attack_detected"])

        self.assertEqual(
            result["execution_status"],
            "defense_error"
        )

        self.assertEqual(llm.calls, 0)

    def test_d5_token_accounting(self):

        pipeline, _, _ = self.build_pipeline()

        result = pipeline.ask(
            TEST_QUESTION,
            defense_mode="D5"
        )

        self.assertEqual(
            result["defense_total_tokens"], 240
        )

        self.assertEqual(
            result["combined_total_tokens"], 256
        )

    def test_ablation_pipeline(self):

        pipeline, _, client = self.build_pipeline()

        result = pipeline.ask(
            TEST_QUESTION,
            defense_mode="D3+D4"
        )

        self.assertTrue(result["isolation_applied"])

        self.assertTrue(
            result["output_validation_applied"]
        )

        self.assertEqual(client.calls, 0)


if __name__ == "__main__":

    unittest.main(verbosity=2)
