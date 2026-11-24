
import json
import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
METADATA = ROOT / "data" / "metadata"

METADATA.mkdir(parents=True, exist_ok=True)


def load_jsonl(filename):
    with (PROCESSED / filename).open(
        "r", encoding="utf-8-sig"
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():

    records = load_jsonl("bipia_review_samples.jsonl")

    usage = Counter(
        r["attack_template_id"]
        for r in records
    )

    unique_templates = {}

    for record in records:

        template_id = record["attack_template_id"]

        if template_id in unique_templates:
            continue

        unique_templates[template_id] = {
            "template_id": template_id,
            "source_category": record["source_category"],
            "source_split": record["source_split"],
            "candidate_family": record["attack_family"],
            "benchmark_template": record["benchmark_template"],
            "sample_count": usage[template_id],
            "attack_goal": "",
            "success_criteria": "",
            "review_status": "pending",
            "review_notes": ""
        }

    rows = list(unique_templates.values())

    rows.sort(
        key=lambda r: (
            r["source_split"],
            r["candidate_family"],
            r["template_id"]
        )
    )

    output = METADATA / "bipia_final_review.csv"

    with output.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 55)
    print("BIPIA FINAL REVIEW SHEET GENERATED")
    print("=" * 55)

    print(f"Total BIPIA records: {len(records)}")
    print(f"Unique templates: {len(rows)}")
    print(f"Templates pending review: {len(rows)}")

    print("\nUnique templates by family:")

    for family, count in Counter(
        r["candidate_family"] for r in rows
    ).items():
        print(f"{family}: {count}")

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
