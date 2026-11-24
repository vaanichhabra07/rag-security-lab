
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

QA_PATH = ROOT / "data" / "corpus" / "qa_pairs.jsonl"

OUTPUT_DIR = ROOT / "data" / "processed"
METADATA_DIR = ROOT / "data" / "metadata"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def build_benign_samples(qa_pairs):

    records = []

    for index, qa in enumerate(qa_pairs, start=1):

        record = {
            "sample_id": f"BENIGN_{index:04d}",
            "qa_id": qa["qa_id"],
            "document_id": qa["document_id"],
            "domain": qa["domain"],
            "question": qa["question"],
            "user_input": qa["question"],
            "clean_context": qa["clean_context"],
            "modified_context": qa["clean_context"],
            "conversation_history": [],
            "label": "benign",
            "attack_family": "none",
            "attack_channel": "none",
            "attack_template_id": None,
            "novelty": "not_applicable",
            "split": "unassigned",
            "expected_answer": qa["expected_answer"],
            "attack_goal": None,
            "success_criteria": None,
            "source": "synthetic_institutional",
            "source_id": qa["qa_id"]
        }

        records.append(record)

    return records


def validate_records(records):

    assert len(records) == 250

    assert len({
        record["sample_id"]
        for record in records
    }) == len(records)

    for record in records:

        assert record["label"] == "benign"

        assert record["user_input"] == record["question"]

        assert (
            record["clean_context"]
            == record["modified_context"]
        )

        assert (
            record["expected_answer"].casefold()
            in record["clean_context"].casefold()
        )

        assert record["attack_template_id"] is None

    print("All benign validation checks passed.")


def save_jsonl(path, records):

    with path.open("w", encoding="utf-8") as file:

        for record in records:

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main():

    qa_pairs = load_jsonl(QA_PATH)

    records = build_benign_samples(qa_pairs)

    validate_records(records)

    output_path = OUTPUT_DIR / "benign_samples.jsonl"

    save_jsonl(output_path, records)

    statistics = {
        "total_samples": len(records),
        "label": "benign",
        "unique_questions": len({
            r["question"] for r in records
        }),
        "unique_documents": len({
            r["document_id"] for r in records
        }),
        "domain_distribution": dict(
            Counter(r["domain"] for r in records)
        )
    }

    stats_path = (
        METADATA_DIR / "benign_statistics.json"
    )

    stats_path.write_text(
        json.dumps(statistics, indent=2),
        encoding="utf-8"
    )

    print("=" * 50)
    print("BENIGN DATASET GENERATION SUCCESSFUL")
    print("=" * 50)

    print(f"Total samples: {len(records)}")
    print(
        f"Unique documents: "
        f"{statistics['unique_documents']}"
    )

    for domain, count in (
        statistics["domain_distribution"].items()
    ):
        print(f"{domain}: {count}")

    print(f"\nDataset saved: {output_path}")
    print(f"Statistics saved: {stats_path}")


if __name__ == "__main__":
    main()
