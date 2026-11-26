
import sys
import json

from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.rag.retriever import Retriever


def read_jsonl(path):

    with path.open(
        encoding="utf-8-sig"
    ) as file:

        return [

            json.loads(line)

            for line in file

            if line.strip()
        ]


def main():

    processed = (
        ROOT / "data" / "v2" / "processed"
    )

    qa_pairs = read_jsonl(

        processed / "expanded_qa_pairs.jsonl"
    )

    plan = read_jsonl(

        processed
        / "adversarial_sample_plan_revised.jsonl"
    )

    training_documents = {

        record["document_id"]

        for record in plan

        if record["split"] == "train"
    }

    development_questions = [

        record for record in qa_pairs

        if record["document_id"]
        in training_documents
    ]

    assert development_questions

    retriever = Retriever()

    results = []

    for index, record in enumerate(

        development_questions,

        start=1
    ):

        retrieval = retriever.retrieve(

            query=record["question"],

            top_k=5
        )

        chunks = retrieval[
            "retrieved_chunks"
        ]

        document_ids = [

            chunk["document_id"]

            for chunk in chunks
        ]

        expected_id = record["document_id"]

        rank = (

            document_ids.index(expected_id) + 1

            if expected_id in document_ids

            else None
        )

        expected_answer = " ".join(

            str(
                record["expected_answer"]
            ).casefold().split()
        )

        evidence_found = any(

            chunk["document_id"] == expected_id

            and expected_answer in " ".join(
                chunk["text"].casefold().split()
            )

            for chunk in chunks
        )

        results.append({

            "qa_id": record["qa_id"],

            "expected_document_id":
                expected_id,

            "retrieved_document_ids":
                document_ids,

            "rank": rank,

            "recall_at_1":
                int(rank is not None and rank <= 1),

            "recall_at_3":
                int(rank is not None and rank <= 3),

            "recall_at_5":
                int(rank is not None and rank <= 5),

            "reciprocal_rank":
                1 / rank if rank else 0,

            "answer_evidence_found":
                evidence_found,

            "latency_ms":
                retrieval["retrieval_latency_ms"]
        })

        if index % 20 == 0:

            print(

                f"Evaluated {index}/"
                f"{len(development_questions)}"
            )

    report = {

        "evaluation_scope":
            "training_document_qa_pairs",

        "questions_evaluated": len(results),

        "recall_at_1": round(

            mean(
                r["recall_at_1"]
                for r in results
            ), 4
        ),

        "recall_at_3": round(

            mean(
                r["recall_at_3"]
                for r in results
            ), 4
        ),

        "recall_at_5": round(

            mean(
                r["recall_at_5"]
                for r in results
            ), 4
        ),

        "mrr_at_5": round(

            mean(
                r["reciprocal_rank"]
                for r in results
            ), 4
        ),

        "answer_evidence_coverage_at_5": round(

            mean(
                r["answer_evidence_found"]
                for r in results
            ), 4
        ),

        "average_retrieval_latency_ms": round(

            mean(
                r["latency_ms"]
                for r in results
            ), 2
        ),

        "formal_test_evaluation": False
    }

    metadata = (
        ROOT / "data" / "v2" / "metadata"
    )

    metadata.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        metadata / "retrieval_evaluation.json"
    ).write_text(

        json.dumps(report, indent=2),

        encoding="utf-8"
    )

    (
        processed / "retrieval_dev_results.jsonl"
    ).write_text(

        "\n".join(

            json.dumps(
                r,
                ensure_ascii=False
            )

            for r in results

        ) + "\n",

        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    print("\nRETRIEVAL EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
