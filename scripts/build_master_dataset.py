
import json
import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
METADATA = ROOT / "data" / "metadata"

FAMILIES = {
    "direct_injection",
    "instruction_manipulation",
    "obfuscation",
    "indirect_injection",
    "multi_stage"
}


def load_jsonl(path):

    with path.open("r", encoding="utf-8-sig") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def save_jsonl(path, records):

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )


def validate(records, plan):

    assert len(records) == 1500

    assert len({
        r["sample_id"] for r in records
    }) == len(records)

    labels = Counter(r["label"] for r in records)

    assert labels["benign"] == 250
    assert labels["adversarial"] == 1250

    attacks = [
        r for r in records
        if r["label"] == "adversarial"
    ]

    planned = {
        r["planned_sample_id"]: r
        for r in plan
    }

    assert {r["sample_id"] for r in attacks} == set(planned)

    families = Counter(
        r["attack_family"] for r in attacks
    )

    assert set(families) == FAMILIES

    for family in FAMILIES:
        assert families[family] == 250

    for record in attacks:

        slot = planned[record["sample_id"]]

        for field in [
            "qa_id", "document_id",
            "attack_family", "split"
        ]:
            assert record[field] == slot[field]

        assert record["attack_template_id"]
        assert record["attack_goal"]
        assert record["success_criteria"]
        assert record["expected_answer"]

        changed = (
            record["user_input"] != record["question"]
            or record["modified_context"]
            != record["clean_context"]
            or bool(record["conversation_history"])
        )

        assert changed, (
            f"No adversarial input: "
            f"{record['sample_id']}"
        )

    print("Master dataset structural validation passed.")


def main():

    required = [
        PROCESSED / "benign_samples.jsonl",
        PROCESSED / "custom_samples.jsonl",
        PROCESSED / "bipia_samples.jsonl"
    ]

    missing = [
        str(path.name)
        for path in required
        if not path.exists()
    ]

    if missing:

        print("Dataset construction is not complete.")

        for name in missing:
            print(f"Pending file: {name}")

        return

    records = []

    for path in required:
        records.extend(load_jsonl(path))

    plan = load_jsonl(
        PROCESSED / "adversarial_sample_plan.jsonl"
    )

    validate(records, plan)

    jsonl_path = PROCESSED / "master_dataset.jsonl"
    csv_path = PROCESSED / "master_dataset.csv"

    save_jsonl(jsonl_path, records)

    fields = sorted({
        key
        for record in records
        for key in record
    })

    with csv_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for record in records:

            row = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
                for key, value in record.items()
            }

            writer.writerow(row)

    statistics = {
        "total_samples": len(records),
        "label_distribution": dict(
            Counter(r["label"] for r in records)
        ),
        "family_distribution": dict(
            Counter(r["attack_family"] for r in records)
        ),
        "source_distribution": dict(
            Counter(r["source"] for r in records)
        ),
        "unique_documents": len({
            r["document_id"] for r in records
        }),
        "structural_validation": "passed"
    }

    (METADATA / "dataset_statistics.json").write_text(
        json.dumps(statistics, indent=2),
        encoding="utf-8"
    )

    print("MASTER DATASET CREATED SUCCESSFULLY")
    print(f"Total samples: {len(records)}")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
