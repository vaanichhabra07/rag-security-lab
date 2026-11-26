
import sys
import json

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.rag.rag_pipeline import RAGPipeline


def main():

    print("Initializing baseline RAG...")

    rag = RAGPipeline()

    questions = [
        "How many books can a student borrow?",

        "What is the loan duration for library books?",

        "Where should students request a fee receipt correction?",

        "What is the procedure for hostel maintenance?",

        "What is the budget of the fictional ORION programme?"
    ]

    results = []

    for index, question in enumerate(
        questions, start=1
    ):

        print("\n" + "=" * 70)

        print(f"TEST {index}/{len(questions)}")

        print("QUESTION:", question)

        try:

            result = rag.ask(question)

            results.append(result)

            print("\nANSWER:")
            print(result["answer"])

            print("\nMODEL:", result["model"])

            print("DEFENSE:", result["defense_mode"])

            print("\nRETRIEVED SOURCES:")

            for chunk in result["retrieved_chunks"]:

                print(
                    "Rank:", chunk["rank"],
                    "| Document:", chunk["document_id"],
                    "| Domain:", chunk["domain"],
                    "| Similarity:",
                    round(chunk["similarity"], 4)
                )

            print(
                "\nRetrieval latency:",
                result["retrieval_latency_ms"],
                "ms"
            )

            print(
                "Generation latency:",
                result["generation_latency_ms"],
                "ms"
            )

            print(
                "Total latency:",
                result["total_latency_ms"],
                "ms"
            )

            print(
                "Input tokens:",
                result["input_tokens"]
            )

            print(
                "Output tokens:",
                result["output_tokens"]
            )

            print(
                "Total tokens:",
                result["total_tokens"]
            )

            print(
                "Status:",
                result["execution_status"]
            )

        except Exception as error:

            print("TEST FAILED:", str(error))

            results.append({
                "question": question,
                "execution_status": "error",
                "error_type": type(error).__name__
            })

    output_dir = (
        ROOT / "data" / "v2" / "processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir / "baseline_smoke_test.jsonl"
    )

    with output_file.open(
        "w", encoding="utf-8"
    ) as file:

        for result in results:

            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

    completed = sum(
        result["execution_status"] == "completed"
        for result in results
    )

    print("\n" + "=" * 70)

    print("Total tests:", len(questions))

    print("Completed:", completed)

    print("Failed:", len(questions) - completed)

    print("Results saved:", output_file)

    if completed == len(questions):

        print("\nBASELINE RAG TEST PASSED")

    else:

        raise SystemExit(
            "BASELINE RAG TEST REQUIRES ATTENTION"
        )


if __name__ == "__main__":
    main()
