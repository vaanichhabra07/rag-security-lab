
import sys
import json
import time

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.rag.rag_pipeline import RAGPipeline


MODES = ["D0", "D1", "D2", "D3", "D4", "D5"]

QUESTION = "How many library books can a student borrow?"

REQUEST_DELAY_SECONDS = 5


def main():

    pipeline = RAGPipeline()

    results = []

    for mode in MODES:

        print("\n" + "=" * 65)

        print("TESTING DEFENSE:", mode)

        try:

            result = pipeline.ask(
                question=QUESTION,
                top_k=1,
                defense_mode=mode
            )

            summary = {
                "defense_mode": mode,
                "answer": result["answer"],
                "active_layers": result["active_layers"],
                "executed_layers": result["executed_layers"],
                "blocked": result["blocked"],
                "blocked_by": result["blocked_by"],
                "isolation_applied":
                    result["isolation_applied"],
                "output_validation_status":
                    result["output_validation_status"],
                "classifier_errors":
                    result["classifier_errors"],
                "defense_api_calls":
                    result["defense_api_calls"],
                "total_latency_ms":
                    result["total_latency_ms"],
                "combined_total_tokens":
                    result["combined_total_tokens"],
                "execution_status":
                    result["execution_status"]
            }

            print(json.dumps(summary, indent=2))

            results.append(summary)

        except Exception as error:

            failure = {
                "defense_mode": mode,
                "execution_status": "error",
                "error_type": type(error).__name__
            }

            results.append(failure)

            print(json.dumps(failure, indent=2))

        if mode != MODES[-1]:

            time.sleep(REQUEST_DELAY_SECONDS)

    output_dir = ROOT / "data/v2/processed"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / "all_defenses_live_test.json"
    )

    output_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    passed = sum(
        row.get("execution_status") == "completed"
        and not row.get("classifier_errors", 0)
        for row in results
    )

    print("\n" + "=" * 65)

    print("Total configurations:", len(MODES))

    print("Completed without classifier errors:", passed)

    print("Results saved:", output_path)

    if passed == len(MODES):

        print("ALL SIX DEFENSES LIVE SMOKE TEST PASSED")

    else:

        raise SystemExit(
            "ONE OR MORE DEFENSES REQUIRE REVIEW"
        )


if __name__ == "__main__":

    main()
