
import json
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

BIPIA_DIR = ROOT / "data" / "raw" / "bipia" / "benchmark"
OUTPUT_DIR = ROOT / "data" / "processed"
METADATA_DIR = ROOT / "data" / "metadata"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_templates():
    records = []

    for split in ["train", "test"]:

        filename = f"text_attack_{split}.json"
        path = BIPIA_DIR / filename

        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(f"Unexpected structure: {filename}")

        for category, templates in data.items():

            if not isinstance(templates, list):
                raise ValueError(
                    f"Unexpected category format: {category}"
                )

            for index, template in enumerate(templates, start=1):

                if not isinstance(template, str):
                    raise TypeError(
                        f"Unexpected template type in {category}: "
                        f"{type(template).__name__}"
                    )

                template = template.strip()

                if not template:
                    raise ValueError("Empty attack template")

                template_hash = hashlib.sha256(
                    template.encode("utf-8")
                ).hexdigest()

                records.append({
                    "template_id": (
                        f"BIPIA_{split.upper()}_"
                        f"{len(records) + 1:04d}"
                    ),
                    "source": "BIPIA",
                    "source_file": filename,
                    "source_split": split,
                    "source_category": category,
                    "source_index": index,
                    "attack_text": template,
                    "template_hash": template_hash
                })

    return records


def save_jsonl(path, records):

    with path.open("w", encoding="utf-8") as file:

        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main():

    records = extract_templates()

    split_counts = Counter(
        record["source_split"]
        for record in records
    )

    train_hashes = {
        record["template_hash"]
        for record in records
        if record["source_split"] == "train"
    }

    test_hashes = {
        record["template_hash"]
        for record in records
        if record["source_split"] == "test"
    }

    overlap = train_hashes.intersection(test_hashes)

    train_categories = sorted({
        record["source_category"]
        for record in records
        if record["source_split"] == "train"
    })

    test_categories = sorted({
        record["source_category"]
        for record in records
        if record["source_split"] == "test"
    })

    statistics = {
        "source": "Microsoft BIPIA",
        "total_templates": len(records),
        "train_templates": split_counts["train"],
        "test_templates": split_counts["test"],
        "unique_templates": len({
            record["template_hash"]
            for record in records
        }),
        "exact_train_test_overlap": len(overlap),
        "train_categories": train_categories,
        "test_categories": test_categories,
        "shared_categories": sorted(
            set(train_categories).intersection(test_categories)
        )
    }

    output_path = OUTPUT_DIR / "bipia_templates.jsonl"

    save_jsonl(output_path, records)

    stats_path = METADATA_DIR / "bipia_template_statistics.json"

    stats_path.write_text(
        json.dumps(statistics, indent=2),
        encoding="utf-8"
    )

    print("=" * 55)
    print("BIPIA TEMPLATE EXTRACTION SUCCESSFUL")
    print("=" * 55)

    print(f"Total templates: {len(records)}")
    print(f"Training templates: {split_counts['train']}")
    print(f"Testing templates: {split_counts['test']}")
    print(f"Unique templates: {statistics['unique_templates']}")
    print(f"Exact train/test overlap: {len(overlap)}")

    print(f"\nTraining categories: {len(train_categories)}")
    print(f"Testing categories: {len(test_categories)}")

    print(f"\nTemplates saved: {output_path}")
    print(f"Statistics saved: {stats_path}")


if __name__ == "__main__":
    main()
