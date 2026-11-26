
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


class RAGPipeline:

    SUPPORTED_MODES = {"D0", "D1"}

    def __init__(
        self,
        provider=None,
        retriever=None,
        llm_client=None
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

        request_id = str(uuid.uuid4())

        result = {

            "request_id": request_id,

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

            "finish_reason": None,

            "attack_detected": False,

            "blocked": False,

            "matched_rules": [],

            "flagged_chunks": [],

            "defense_details": {},

            "llm_called": False,

            "execution_status": "pending"
        }

        def finish():

            result["total_latency_ms"] = round(

                (
                    time.perf_counter()
                    - start
                ) * 1000,

                2
            )

            return result

        # ----------------------------------
        # D1: USER INPUT SCANNING
        # ----------------------------------

        if defense_mode == "D1":

            defense_start = time.perf_counter()

            input_scan = self.rule_defense.scan(

                text=question,

                channel="user_input"
            )

            result["defense_details"][
                "input_scan"
            ] = input_scan

            result["defense_latency_ms"] += (

                time.perf_counter()
                - defense_start

            ) * 1000

            if input_scan["detected"]:

                result["attack_detected"] = True

                result["blocked"] = True

                result["matched_rules"] = (
                    input_scan["matched_rules"]
                )

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

        # ----------------------------------
        # RETRIEVAL
        # ----------------------------------

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

        # ----------------------------------
        # D1: RETRIEVED CONTEXT SCANNING
        # ----------------------------------

        if defense_mode == "D1":

            defense_start = time.perf_counter()

            context_scan = (
                self.rule_defense.scan_context(
                    chunks
                )
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
                        scan["matched_rules"]
                    )

            result["defense_latency_ms"] += (

                time.perf_counter()
                - defense_start

            ) * 1000

            if chunks and not context_chunks:

                result["blocked"] = True

                result["answer"] = (
                    "The retrieved information "
                    "was flagged by the security "
                    "policy. A reliable answer "
                    "cannot be provided."
                )

                result["execution_status"] = (
                    "blocked_by_defense"
                )

                return finish()

        result["context_used"] = context_chunks

        # ----------------------------------
        # PROMPT CONSTRUCTION
        # ----------------------------------

        prompt = build_baseline_prompt(

            question=question,

            retrieved_chunks=context_chunks
        )

        # ----------------------------------
        # LLM GENERATION
        # ----------------------------------

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

        result["defense_latency_ms"] = round(
            result["defense_latency_ms"], 4
        )

        return finish()
