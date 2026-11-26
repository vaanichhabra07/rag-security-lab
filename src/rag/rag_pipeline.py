
import os
import time
import uuid

from src.rag.retriever import Retriever
from src.rag.llm_client import LLMClient
from src.rag.prompt_builder import build_baseline_prompt


class RAGPipeline:

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

        self.default_top_k = int(
            os.getenv("RAG_TOP_K", "3")
        )

    def ask(
        self,
        question,
        top_k=None,
        domain=None
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

        start = time.perf_counter()

        request_id = str(uuid.uuid4())

        if top_k is None:
            top_k = self.default_top_k

        # Step 1: Retrieve relevant documents

        retrieval = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            domain=domain
        )

        chunks = retrieval["retrieved_chunks"]

        # Step 2: Construct the RAG prompt

        prompt = build_baseline_prompt(
            question=question,
            retrieved_chunks=chunks
        )

        # Step 3: Generate the answer

        generation = self.llm_client.generate(
            system_prompt=prompt["system_prompt"],
            user_prompt=prompt["user_prompt"]
        )

        total_latency = (
            time.perf_counter() - start
        ) * 1000

        # Step 4: Return answer and diagnostics

        return {
            "request_id": request_id,

            "defense_mode": "D0",

            "question": question,

            "answer": generation["answer"],

            "provider": generation["provider"],

            "model": generation["model"],

            "top_k": top_k,

            "retrieved_chunks": chunks,

            "retrieval_latency_ms":
                retrieval["retrieval_latency_ms"],

            "generation_latency_ms":
                generation["generation_latency_ms"],

            "total_latency_ms":
                round(total_latency, 2),

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
        }
