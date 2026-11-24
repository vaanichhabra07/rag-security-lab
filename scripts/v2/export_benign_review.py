
import json
import csv

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "data" / "v2" / "processed"
    / "benign_samples.jsonl"
)

OUTPUT = (
    ROOT / "data" / "v2" / "metadata"
    / "benign_review.csv"
)

with INPUT.open(encoding="utf-8") as f:

    records = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

fields = [
    "sample_id",
    "qa_id",
    "document_id",
    "domain",
    "benign_category",
    "language",
    "user_input",
    "conversation_history",
    "expected_answer",
    "expected_behavior",
    "answerable",
    "annotation_status",
    "review_notes"
]

with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    for record in records:

        row = {
            field: record.get(field, "")
            for field in fields
        }

        row["conversation_history"] = json.dumps(
            record["conversation_history"],
            ensure_ascii=False
        )

        row["review_notes"] = ""

        writer.writerow(row)

print(f"Exported {len(records)} samples")

print(f"Review file: {OUTPUT}")
