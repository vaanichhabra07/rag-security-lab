
import os
import time
import uuid

from src.rag.retriever import Retriever
from src.rag.llm_client import LLMClient

from src.rag.prompt_builder import (
    build_baseline_prompt
)

from src.defenses.rule_based import (
    RuleBasedDefense
)

from src.defenses.llm_detector import (
    LLMDetector
)


class RAGPipeline:

    SUPPORTED_MODES = {"D0", "D1", "D2"}

    def __init__(
        self,
        provider=None,
        retriever=None,
        llm_client=None,
        llm_detector=None
    ):

        self.retriever = (
            retriever
            if retriever is not None
            else Retriever()
        )

        self.llm_client = (
            llm_client
            if llm_client is not None
            else LLMClient(provider=provider)
        )

        self.rule_defense = RuleBasedDefense()

        self.llm_detector = (
            llm_detector
            if llm_detector is not None
            else LLMDetector()
        )

        self.default_top_k = int(
            os.getenv("RAG_TOP_K", "3")
        )

    def ask(
        self,
        question,
        top_k=None,
        domain=None,
        defense_mode="D0"
    ):

        if not isinstance(question, str):

            raise TypeError(
                "Question must be a string"
            )

        question = question.strip()

        if not question:

            raise ValueError(
                "Question cannot be empty"
            )

        defense_mode = defense_mode.upper()

        if defense_mode not in self.SUPPORTED_MODES:

            raise ValueError(
                f"Unsupported defense: {defense_mode}"
            )

        if top_k is None:

            top_k = self.default_top_k

        if top_k <= 0:

            raise ValueError(
                "top_k must be positive"
            )

        start = time.perf_counter()

        result = {

            "request_id": str(uuid.uuid4()),

            "defense_mode": defense_mode,

            "question": question,

            "answer": "",

            "provider": self.llm_client.provider,

            "model": self.llm_client.model,

            "top_k": top_k,

            "retrieved_chunks": [],

            "context_used": [],

            "retrieval_latency_ms": 0,

            "generation_latency_ms": 0,

            "defense_latency_ms": 0,

            "total_latency_ms": 0,

            "input_tokens": 0,

            "output_tokens": 0,

            "total_tokens": 0,

            "defense_input_tokens": 0,

            "defense_output_tokens": 0,

            "defense_total_tokens": 0,

            "defense_api_calls": 0,

            "classifier_errors": 0,

            "classifier_results": [],

            "classifier_model": (
                self.llm_detector.model
                if defense_mode == "D2"
                else None
            ),

            "finish_reason": None,

            "attack_detected": False,

            "blocked": False,

            "matched_rules": [],

            "flagged_chunks": [],

            "defense_details": {},

            "llm_called": False,

            "execution_status": "pending"
        }

        def add_tokens(current, new):

            if current is None or new is None:

                return None

            return current + new

        def record_classifier(scan):

            result["classifier_results"].append(
                scan
            )

            result["defense_api_calls"] += 1

            for key in [

                "input_tokens",

                "output_tokens",

                "total_tokens"
            ]:

                target = "defense_" + key

                result[target] = add_tokens(

                    result[target],

                    scan.get(key)
                )

            if scan["classification_status"] == "error":

                result["classifier_errors"] += 1

        def finish():

            result["total_latency_ms"] = round(

                (
                    time.perf_counter() - start
                ) * 1000,

                2
            )

            result["defense_latency_ms"] = round(

                result["defense_latency_ms"],

                4
            )

            result["combined_total_tokens"] = (
                add_tokens(

                    result["total_tokens"],

                    result["defense_total_tokens"]
                )
            )

            return result

        # ------------------------------------------
        # SELECT DETECTOR
        # ------------------------------------------

        detector = None

        if defense_mode == "D1":

            detector = self.rule_defense

        elif defense_mode == "D2":

            detector = self.llm_detector

        # ------------------------------------------
        # INPUT SECURITY SCANNING
        # ------------------------------------------

        if detector is not None:

            defense_start = time.perf_counter()

            input_scan = detector.scan(

                text=question,

                channel="user_input"
            )

            result["defense_details"][
                "input_scan"
            ] = input_scan

            if defense_mode == "D2":

                record_classifier(input_scan)

            result["defense_latency_ms"] += (

                time.perf_counter()
                - defense_start

            ) * 1000

            if input_scan["detected"] is True:

                result["attack_detected"] = True

            result["matched_rules"].extend(

                input_scan.get(
                    "matched_rules", []
                )
            )

            if input_scan["action"] == "block":

                result["blocked"] = True

                if input_scan.get(
                    "classification_status"
                ) == "error":

                    result["answer"] = (
                        "The security classifier "
                        "is temporarily unavailable. "
                        "Please try again later."
                    )

                    result["execution_status"] = (
                        "defense_error"
                    )

                else:

                    result["answer"] = (
                        "Your request was flagged by "
                        "the security policy. Please "
                        "rephrase your institutional "
                        "question."
                    )

                    result["execution_status"] = (
                        "blocked_by_defense"
                    )

                return finish()

        # ------------------------------------------
        # DOCUMENT RETRIEVAL
        # ------------------------------------------

        retrieval = self.retriever.retrieve(

            query=question,

            top_k=top_k,

            domain=domain
        )

        chunks = retrieval["retrieved_chunks"]

        result["retrieved_chunks"] = chunks

        result["retrieval_latency_ms"] = (
            retrieval["retrieval_latency_ms"]
        )

        context_chunks = chunks

        # ------------------------------------------
        # CONTEXT SECURITY SCANNING
        # ------------------------------------------

        if detector is not None:

            defense_start = time.perf_counter()

            context_scan = detector.scan_context(
                chunks
            )

            context_chunks = (
                context_scan["allowed_chunks"]
            )

            result["flagged_chunks"] = (
                context_scan["flagged_chunks"]
            )

            result["defense_details"][
                "context_scan"
            ] = context_scan["scan_results"]

            if context_scan["detected"]:

                result["attack_detected"] = True

            for scan in context_scan[
                "scan_results"
            ]:

                result["matched_rules"].extend(

                    scan.get(
                        "matched_rules", []
                    )
                )

                if defense_mode == "D2":

                    record_classifier(scan)

            result["defense_latency_ms"] += (

                time.perf_counter()
                - defense_start

            ) * 1000

            if chunks and not context_chunks:

                result["blocked"] = True

                result["answer"] = (
                    "A reliable answer cannot be "
                    "provided because the retrieved "
                    "context was excluded by the "
                    "security policy."
                )

                result["execution_status"] = (

                    "defense_error"

                    if result["classifier_errors"]

                    else "blocked_by_defense"
                )

                return finish()

        result["context_used"] = context_chunks

        # ------------------------------------------
        # PROMPT CONSTRUCTION
        # ------------------------------------------

        prompt = build_baseline_prompt(

            question=question,

            retrieved_chunks=context_chunks
        )

        # ------------------------------------------
        # ANSWER GENERATION
        # ------------------------------------------

        result["llm_called"] = True

        generation = self.llm_client.generate(

            system_prompt=prompt["system_prompt"],

            user_prompt=prompt["user_prompt"]
        )

        result.update({

            "answer": generation["answer"],

            "provider": generation["provider"],

            "model": generation["model"],

            "generation_latency_ms":
                generation["generation_latency_ms"],

            "input_tokens":
                generation["input_tokens"],

            "output_tokens":
                generation["output_tokens"],

            "total_tokens":
                generation["total_tokens"],

            "finish_reason":
                generation["finish_reason"],

            "execution_status": (

                "completed"

                if generation["answer"].strip()

                else "empty_response"
            )
        })

        return finish()
