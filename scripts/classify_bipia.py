
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "data/processed/bipia_templates.jsonl"
)

OUTPUT_PATH = (
    ROOT / "data/processed/bipia_classified.jsonl"
)

STATS_PATH = (
    ROOT / "data/metadata/bipia_classification_statistics.json"
)

OBFUSCATION_CATEGORIES = {
    "Alphanumeric Substitution",
    "Homophonic Substitution",
    "Misspelling Intentionally",
    "Anagramming",
    "Space Removal & Grouping",
    "Substitution Ciphers",
    "Base Encoding",
    "Reverse Text",
    "Emoji Substitution"
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def classify_template(record):

    category = record["source_category"]

    # BIPIA's original delivery mechanism is indirect.
    family = "indirect_injection"

    candidate_family = family

    if category in OBFUSCATION_CATEGORIES:
        candidate_family = "obfuscation"

    elif category == "Instruction":
        candidate_family = "instruction_manipulation"

    return {
        **record,
        "original_attack_channel": "retrieved_context",
        "original_attack_family": family,
        "candidate_attack_family": candidate_family,
        "classification_status": "requires_review",
        "template_partition": record["source_split"]
    }


def main():

    templates = load_jsonl(INPUT_PATH)

    classified = [
        classify_template(record)
        for record in templates
    ]

    distribution = Counter(
        record["candidate_attack_family"]
        for record in classified
    )

    split_distribution = Counter(
        (
            record["template_partition"],
            record["candidate_attack_family"]
        )
        for record in classified
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for record in classified:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    statistics = {
        "total_templates": len(classified),
        "candidate_family_distribution": dict(distribution),
        "partition_distribution": {
            f"{split}_{family}": count
            for (split, family), count
            in split_distribution.items()
        },
        "classification_method": "category_based_preliminary",
        "manual_review_required": True
    }

    STATS_PATH.write_text(
        json.dumps(statistics, indent=2),
        encoding="utf-8"
    )

    assert len(classified) == len(templates)

    print("=" * 55)
    print("BIPIA PRELIMINARY CLASSIFICATION SUCCESSFUL")
    print("=" * 55)

    print(f"Total templates: {len(classified)}")

    for family, count in distribution.items():
        print(f"{family}: {count}")

    print("\nPartition distribution:")

    for (split, family), count in split_distribution.items():
        print(f"{split} | {family}: {count}")

    print(f"\nOutput saved: {OUTPUT_PATH}")
    print(f"Statistics saved: {STATS_PATH}")


if __name__ == "__main__":
    main()
