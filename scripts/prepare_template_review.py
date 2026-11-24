
import json
import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
METADATA = ROOT / "data" / "metadata"


def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():

    bipia = load_jsonl(
        PROCESSED / "bipia_classified.jsonl"
    )

    custom = load_jsonl(
        PROCESSED / "custom_templates.jsonl"
    )

    rows = []

    for record in bipia:

        rows.append({
            "template_id": record["template_id"],
            "source": "BIPIA",
            "source_split": record["source_split"],
            "category": record["source_category"],
            "candidate_family":
                record["candidate_attack_family"],
            "attack_channel": "retrieved_context",
            "review_status": "pending",
            "success_criterion": "",
            "review_notes": ""
        })

    for record in custom:

        rows.append({
            "template_id": record["template_id"],
            "source": "custom",
            "source_split": record["source_split"],
            "category": record["scenario"],
            "candidate_family": record["attack_family"],
            "attack_channel": record["attack_channel"],
            "review_status": "pending_materialization",
            "success_criterion": "synthetic_marker",
            "review_notes": ""
        })

    output = METADATA / "template_review.csv"

    with output.open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    assert len(rows) == 200

    print("=" * 50)
    print("TEMPLATE REVIEW MANIFEST CREATED")
    print("=" * 50)

    print(f"Total templates: {len(rows)}")

    for source, count in Counter(
        r["source"] for r in rows
    ).items():
        print(f"{source}: {count}")

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
