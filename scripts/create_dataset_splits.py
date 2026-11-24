
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
METADATA = ROOT / "data" / "metadata"

SPLITS.mkdir(parents=True, exist_ok=True)

EXPECTED_COUNTS = {
    "train": 600,
    "validation": 300,
    "test_seen": 250,
    "test_unseen": 250,
    "test_benign": 100
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )


def main():

    records = load_jsonl(
        PROCESSED / "master_dataset.jsonl"
    )

    assert len(records) == 1500

    # Recover document partitions from the
    # existing adversarial sample assignments.

    document_groups = defaultdict(set)

    for record in records:

        if record["label"] == "benign":
            continue

        split = record["split"]

        if split in ("test_seen", "test_unseen"):
            document_partition = "test"
        else:
            document_partition = split

        document_groups[
            record["document_id"]
        ].add(document_partition)

    # Ensure each document belongs to only
    # one train/validation/test partition.

    for document_id, partitions in document_groups.items():

        assert len(partitions) == 1, (
            f"Document leakage: {document_id}, "
            f"{partitions}"
        )

    document_lookup = {
        document_id: next(iter(partitions))
        for document_id, partitions
        in document_groups.items()
    }

    assert len(document_lookup) == 50

    split_records = defaultdict(list)

    for record in records:

        if record["label"] == "benign":

            partition = document_lookup[
                record["document_id"]
            ]

            record["split"] = (
                "test_benign"
                if partition == "test"
                else partition
            )

            record["novelty"] = "not_applicable"

        split = record["split"]

        assert split in EXPECTED_COUNTS, (
            f"Invalid split: {split}"
        )

        split_records[split].append(record)

    # Validate all expected split counts.

    for split, expected in EXPECTED_COUNTS.items():

        actual = len(split_records[split])

        assert actual == expected, (
            f"{split}: expected {expected}, "
            f"found {actual}"
        )

    all_ids = [
        record["sample_id"]
        for group in split_records.values()
        for record in group
    ]

    assert len(all_ids) == 1500
    assert len(set(all_ids)) == 1500

    # Export individual dataset partitions.

    statistics = {}

    for split, group in split_records.items():

        save_jsonl(
            SPLITS / f"{split}.jsonl",
            group
        )

        statistics[split] = {
            "total_samples": len(group),
            "label_distribution": dict(Counter(
                r["label"] for r in group
            )),
            "family_distribution": dict(Counter(
                r["attack_family"] for r in group
            )),
            "unique_documents": len({
                r["document_id"] for r in group
            }),
            "unique_questions": len({
                r["qa_id"] for r in group
            })
        }

    # Update master JSONL with benign assignments.

    save_jsonl(
        PROCESSED / "master_dataset.jsonl",
        records
    )

    # Keep the master CSV synchronized.

    fieldnames = sorted({
        key
        for record in records
        for key in record
    })

    with (
        PROCESSED / "master_dataset.csv"
    ).open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for record in records:

            row = {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in record.items()
            }

            writer.writerow(row)

    # Save split statistics.

    with (
        METADATA / "split_statistics.json"
    ).open("w", encoding="utf-8") as f:

        json.dump(
            statistics,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("=" * 55)
    print("DATASET SPLITTING COMPLETE")
    print("=" * 55)

    for split in EXPECTED_COUNTS:

        group = split_records[split]

        benign = sum(
            r["label"] == "benign"
            for r in group
        )

        adversarial = len(group) - benign

        print(
            f"{split}: {len(group)} samples "
            f"(benign={benign}, "
            f"adversarial={adversarial})"
        )

    print("\nTotal samples: 1500")
    print("Duplicate sample IDs: 0")
    print("Document partition leakage: 0")

    print("\nDocument distribution:")

    for partition, count in Counter(
        document_lookup.values()
    ).items():

        print(f"{partition}: {count}")

    print("\nMaster dataset updated.")
    print("Split statistics saved.")
    print(f"Dataset splits saved to: {SPLITS}")


if __name__ == "__main__":
    main()
