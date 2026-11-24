
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
METADATA = DATA / "metadata"

SEED = 42

FAMILIES = [
    "direct_injection",
    "instruction_manipulation",
    "obfuscation",
    "indirect_injection",
    "multi_stage"
]


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


def assign_document_groups(benign):

    rng = random.Random(SEED)

    domains = defaultdict(set)

    for record in benign:
        domains[record["domain"]].add(
            record["document_id"]
        )

    assignments = {}

    for domain, document_ids in sorted(domains.items()):

        ids = sorted(document_ids)
        rng.shuffle(ids)

        assert len(ids) == 10

        for index, document_id in enumerate(ids):

            if index < 4:
                group = "train"
            elif index < 6:
                group = "validation"
            else:
                group = "test"

            assignments[document_id] = group

    return assignments


def build_sample_plan(benign, assignments):

    plan = []
    counter = 0

    test_records = defaultdict(list)

    for record in benign:

        group = assignments[record["document_id"]]

        if group == "test":
            test_records[record["domain"]].append(
                record
            )
            continue

        for family in FAMILIES:

            counter += 1

            plan.append({
                "planned_sample_id": f"ADV_{counter:05d}",
                "qa_id": record["qa_id"],
                "document_id": record["document_id"],
                "domain": record["domain"],
                "attack_family": family,
                "split": group,
                "template_partition": "train",
                "status": "pending_materialization"
            })

    # Use the same 50 legitimate questions for
    # seen and unseen attack evaluation.

    rng = random.Random(SEED)

    for domain, records in sorted(test_records.items()):

        selected = rng.sample(records, 10)

        for record in selected:

            for family in FAMILIES:

                for split, partition in [
                    ("test_seen", "train"),
                    ("test_unseen", "test")
                ]:

                    counter += 1

                    plan.append({
                        "planned_sample_id":
                            f"ADV_{counter:05d}",
                        "qa_id": record["qa_id"],
                        "document_id":
                            record["document_id"],
                        "domain": domain,
                        "attack_family": family,
                        "split": split,
                        "template_partition": partition,
                        "status":
                            "pending_materialization"
                    })

    return plan


def main():

    benign = load_jsonl(
        PROCESSED / "benign_samples.jsonl"
    )

    bipia = load_jsonl(
        PROCESSED / "bipia_classified.jsonl"
    )

    custom = load_jsonl(
        PROCESSED / "custom_templates.jsonl"
    )

    assert len(benign) == 250
    assert len(bipia) == 150
    assert len(custom) == 50

    assignments = assign_document_groups(benign)

    plan = build_sample_plan(
        benign,
        assignments
    )

    assert len(assignments) == 50
    assert len(plan) == 1250

    split_counts = Counter(
        r["split"] for r in plan
    )

    family_counts = Counter(
        r["attack_family"] for r in plan
    )

    assert dict(split_counts) == {
        "train": 500,
        "validation": 250,
        "test_seen": 250,
        "test_unseen": 250
    }

    assert all(
        count == 250
        for count in family_counts.values()
    )

    save_jsonl(
        PROCESSED / "adversarial_sample_plan.jsonl",
        plan
    )

    (METADATA / "document_assignments.json").write_text(
        json.dumps(assignments, indent=2),
        encoding="utf-8"
    )

    statistics = {
        "random_seed": SEED,
        "planned_adversarial_samples": len(plan),
        "benign_samples": len(benign),
        "target_master_samples": len(plan) + len(benign),
        "bipia_templates_available": len(bipia),
        "custom_scenarios_available": len(custom),
        "document_distribution": dict(
            Counter(assignments.values())
        ),
        "adversarial_split_distribution":
            dict(split_counts),
        "attack_family_distribution":
            dict(family_counts),
        "status": "planned_not_materialized"
    }

    (METADATA / "benchmark_plan_statistics.json").write_text(
        json.dumps(statistics, indent=2),
        encoding="utf-8"
    )

    print("=" * 55)
    print("BENCHMARK SAMPLE ALLOCATION SUCCESSFUL")
    print("=" * 55)

    print(f"Benign samples: {len(benign)}")
    print(f"Planned adversarial samples: {len(plan)}")
    print(f"Target master dataset: {len(plan) + len(benign)}")

    print("\nAttack family distribution:")

    for family, count in family_counts.items():
        print(f"{family}: {count}")

    print("\nAdversarial partitions:")

    for split, count in split_counts.items():
        print(f"{split}: {count}")

    print("\nDocument groups:")

    for group, count in Counter(
        assignments.values()
    ).items():
        print(f"{group}: {count}")

    print("\nSample allocation saved successfully.")


if __name__ == "__main__":
    main()
