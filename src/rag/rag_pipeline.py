
import os
import time
import uuid

from src.rag.retriever import Retriever
from src.rag.llm_client import LLMClient
from src.rag.prompt_builder import build_baseline_prompt

from src.defenses.rule_based import RuleBasedDefense
from src.defenses.llm_detector import LLMDetector
from src.defenses.context_isolation import ContextIsolationDefense
from src.defenses.output_validator import OutputValidator
from src.defenses.combined_defense import CombinedDefense


class RAGPipeline:

    SUPPORTED_MODES = CombinedDefense.supported_modes()

    def __init__(
        self,
        provider=None,
        retriever=None,
        llm_client=None,
        llm_detector=None,
        output_validator=None
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

        self.context_defense = ContextIsolationDefense()

        self.output_validator = (
            output_validator
            if output_validator is not None
            else OutputValidator()
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
            raise TypeError("Question must be a string")

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty")

        config = CombinedDefense.get_configuration(
            defense_mode
        )

        defense_mode = config.mode

        if top_k is None:
            top_k = self.default_top_k

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        start = time.perf_counter()

        result = {
            "request_id": str(uuid.uuid4()),
            "defense_mode": defense_mode,
            "active_layers": config.active_layers(),
            "executed_layers": [],
            "question": question,
            "answer": "",
            "provider": self.llm_client.provider,
            "model": self.llm_client.model,
            "top_k": top_k,

            "retrieved_chunks": [],
            "context_used": [],
            "flagged_chunks": [],

            "retrieval_latency_ms": 0,
            "generation_latency_ms": 0,
            "defense_latency_ms": 0,
            "isolation_latency_ms": 0,
            "validation_latency_ms": 0,
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
                if config.llm_detection
                else None
            ),

            "attack_detected": False,
            "blocked": False,
            "blocked_stage": None,
            "blocked_by": None,
            "matched_rules": [],
            "defense_details": {},

            "isolation_applied": False,
            "output_validation_applied": False,
            "output_validation_status": "not_evaluated",
            "output_violations": [],

            "llm_called": False,
            "finish_reason": None,
            "execution_status": "pending"
        }

        # ==========================================
        # HELPER FUNCTIONS
        # ==========================================

        def add_tokens(current, new):

            if current is None or new is None:
                return None

            return current + new

        def mark_executed(layer):

            if layer not in result["executed_layers"]:
                result["executed_layers"].append(layer)

        def record_classifier(scan):

            result["classifier_results"].append(scan)

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

            if scan.get("classification_status") == "error":
                result["classifier_errors"] += 1

        def finish():

            result["total_latency_ms"] = round(
                (time.perf_counter() - start) * 1000,
                2
            )

            result["defense_latency_ms"] = round(
                result["defense_latency_ms"],
                4
            )

            result["combined_total_tokens"] = add_tokens(
                result["total_tokens"],
                result["defense_total_tokens"]
            )

            return result

        def block_request(layer, stage, error=False):

            result["blocked"] = True

            result["blocked_by"] = layer

            result["blocked_stage"] = stage

            if error:

                result["answer"] = (
                    "The security classifier is "
                    "temporarily unavailable. "
                    "Please try again later."
                )

                result["execution_status"] = (
                    "defense_error"
                )

            else:

                result["answer"] = (
                    "The request could not be completed "
                    "under the application security policy. "
                    "Please rephrase your institutional "
                    "question."
                )

                result["execution_status"] = (
                    "blocked_by_defense"
                )

            return finish()

        # ==========================================
        # SELECT ACTIVE DETECTORS
        # ==========================================

        detectors = []

        if config.rule_based:

            detectors.append(
                ("D1", self.rule_defense)
            )

        if config.llm_detection:

            detectors.append(
                ("D2", self.llm_detector)
            )

        # ==========================================
        # INPUT SECURITY SCANNING
        # ==========================================

        for layer, detector in detectors:

            mark_executed(layer)

            defense_start = time.perf_counter()

            scan = detector.scan(
                text=question,
                channel="user_input"
            )

            result["defense_latency_ms"] += (
                time.perf_counter() - defense_start
            ) * 1000

            result["defense_details"].setdefault(
                layer, {}
            )["input_scan"] = scan

            if layer == "D2":
                record_classifier(scan)

            if scan.get("detected") is True:
                result["attack_detected"] = True

            result["matched_rules"].extend(
                scan.get("matched_rules", [])
            )

            if scan.get("action") == "block":

                return block_request(
                    layer=layer,
                    stage="user_input",
                    error=(
                        scan.get("classification_status")
                        == "error"
                    )
                )

        # ==========================================
        # DOCUMENT RETRIEVAL
        # ==========================================

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

        context_chunks = list(chunks)

        # ==========================================
        # CONTEXT SECURITY SCANNING
        # ==========================================

        for layer, detector in detectors:

            mark_executed(layer)

            defense_start = time.perf_counter()

            context_scan = detector.scan_context(
                context_chunks
            )

            result["defense_latency_ms"] += (
                time.perf_counter() - defense_start
            ) * 1000

            context_chunks = (
                context_scan["allowed_chunks"]
            )

            result["flagged_chunks"].extend(
                context_scan["flagged_chunks"]
            )

            result["defense_details"].setdefault(
                layer, {}
            )["context_scan"] = (
                context_scan["scan_results"]
            )

            if context_scan.get("detected"):

                result["attack_detected"] = True

            for scan in context_scan["scan_results"]:

                result["matched_rules"].extend(
                    scan.get("matched_rules", [])
                )

                if layer == "D2":
                    record_classifier(scan)

            if chunks and not context_chunks:

                return block_request(
                    layer=layer,
                    stage="retrieved_context",
                    error=bool(
                        context_scan.get(
                            "classifier_errors", 0
                        )
                    )
                )

        # ==========================================
        # CONTEXT AVAILABILITY
        # ==========================================

        result["context_used"] = context_chunks

        if not context_chunks:

            result["answer"] = (
                "The requested information is not "
                "available in the institutional corpus."
            )

            result["execution_status"] = "no_context"

            return finish()

        # ==========================================
        # D3 CONTEXT ISOLATION
        # ==========================================

        if config.context_isolation:

            mark_executed("D3")

            defense_start = time.perf_counter()

            prompt = self.context_defense.build_prompt(
                question=question,
                retrieved_chunks=context_chunks
            )

            result["isolation_applied"] = True

            result["isolation_latency_ms"] = (
                prompt["isolation_latency_ms"]
            )

            result["defense_latency_ms"] += (
                time.perf_counter() - defense_start
            ) * 1000

            result["defense_details"]["D3"] = {
                "isolation_applied": True,
                "isolation_version":
                    prompt["isolation_version"],
                "document_count":
                    prompt["document_count"]
            }

        else:

            prompt = build_baseline_prompt(
                question=question,
                retrieved_chunks=context_chunks
            )

        # ==========================================
        # LLM ANSWER GENERATION
        # ==========================================

        result["llm_called"] = True

        generation = self.llm_client.generate(
            system_prompt=prompt["system_prompt"],
            user_prompt=prompt["user_prompt"]
        )

        raw_answer = generation["answer"]

        result.update({
            "answer": raw_answer,
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
                if raw_answer.strip()
                else "empty_response"
            )
        })

        # ==========================================
        # D4 OUTPUT VALIDATION
        # ==========================================

        if config.output_validation:

            mark_executed("D4")

            defense_start = time.perf_counter()

            validation = self.output_validator.validate(
                answer=raw_answer,
                system_prompt=prompt["system_prompt"]
            )

            result["defense_latency_ms"] += (
                time.perf_counter() - defense_start
            ) * 1000

            result["output_validation_applied"] = True

            result["validation_latency_ms"] = (
                validation["validation_latency_ms"]
            )

            result["output_validation_status"] = (
                validation["validation_status"]
            )

            result["output_violations"] = (
                validation["violations"]
            )

            result["defense_details"]["D4"] = {
                key: value
                for key, value in validation.items()
                if key != "answer"
            }

            result["answer"] = validation["answer"]

            if validation["blocked"]:

                result["attack_detected"] = True

                result["blocked"] = True

                result["blocked_by"] = "D4"

                result["blocked_stage"] = (
                    "generated_output"
                )

                result["execution_status"] = (
                    "blocked_by_output_validator"
                )

        return finish()
