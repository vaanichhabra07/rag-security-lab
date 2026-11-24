
import json
import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
METADATA = ROOT / "data" / "metadata"


def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():

    expected_files = {
        "benign_samples.jsonl": 250,
        "bipia_templates.jsonl": 150,
        "bipia_classified.jsonl": 150,
        "custom_templates.jsonl": 50,
        "adversarial_sample_plan.jsonl": 1250
    }

    print("=" * 55)
    print("BENCHMARK READINESS CHECK")
    print("=" * 55)

    for filename, expected in expected_files.items():

        path = PROCESSED / filename

        assert path.exists(), f"Missing file: {filename}"

        records = load_jsonl(path)

        assert len(records) == expected, (
            f"Unexpected record count: {filename}"
        )

        print(f"PASS: {filename}: {len(records)}")

    plan = load_jsonl(
        PROCESSED / "adversarial_sample_plan.jsonl"
    )

    print("\nPlanned family distribution:")

    for family, count in Counter(
        r["attack_family"] for r in plan
    ).items():
        print(f"{family}: {count}")

    review_path = METADATA / "template_review.csv"

    with review_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reviews = list(csv.DictReader(f))

    assert len(reviews) == 200

    print("\nTemplate review status:")

    for status, count in Counter(
        r["review_status"] for r in reviews
    ).items():
        print(f"{status}: {count}")

    print("\nFinal dataset files:")

    for filename in [
        "custom_samples.jsonl",
        "bipia_samples.jsonl",
        "master_dataset.jsonl",
        "master_dataset.csv"
    ]:
        path = PROCESSED / filename

        if path.exists():
            print(f"AVAILABLE: {filename}")
        else:
            print(f"PENDING: {filename}")

    print("\nAll intermediate file checks passed.")


if __name__ == "__main__":
    main()
