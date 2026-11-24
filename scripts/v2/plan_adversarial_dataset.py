
import json
import random

from pathlib import Path
from collections import Counter, defaultdict


ROOT = Path(__file__).resolve().parents[2]

CONFIG = ROOT / "configs" / "adversarial_allocation_v2.json"

MASTER_CONFIG = ROOT / "configs" / "dataset_v2.json"

PROCESSED = ROOT / "data" / "v2" / "processed"

METADATA = ROOT / "data" / "v2" / "metadata"


def load_json(path):

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_jsonl(path):

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def save_jsonl(path, records):

    with path.open("w", encoding="utf-8") as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


def save_json(path, data):

    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8"
    )


def partition_documents(qa_records, rng):

    by_domain = defaultdict(set)

    for qa in qa_records:

        by_domain[qa["domain"]].add(
            qa["document_id"]
        )

    assignments = {}

    for domain in sorted(by_domain):

        documents = sorted(by_domain[domain])

        rng.shuffle(documents)

        n = len(documents)

        assert n % 5 == 0, (
            f"Unexpected domain size: {domain}: {n}"
        )

        train_end = n * 40 // 100

        val_end = train_end + n * 20 // 100

        for index, document_id in enumerate(documents):

            if index < train_end:

                partition = "train"

            elif index < val_end:

                partition = "validation"

            else:

                partition = "test"

            assert document_id not in assignments

            assignments[document_id] = partition

    return assignments


def select_anchors(pool, count, rng):

    if not pool:

        raise ValueError("Empty QA pool")

    selected = []

    while len(selected) < count:

        batch = list(pool)

        rng.shuffle(batch)

        remaining = count - len(selected)

        selected.extend(batch[:remaining])

    return selected


def main():

    config = load_json(CONFIG)

    master = load_json(MASTER_CONFIG)

    rng = random.Random(config["random_seed"])

    qa_records = load_jsonl(
        PROCESSED / "expanded_qa_pairs.jsonl"
    )

    benign_records = load_jsonl(
        PROCESSED / "benign_samples.jsonl"
    )

    assert len(qa_records) == 500

    assert len(benign_records) == 700

    assert len({
        q["qa_id"] for q in qa_records
    }) == 500

    allocation = config["family_source_allocation"]

    source_totals = Counter()

    for family, sources in allocation.items():

        assert sum(sources.values()) == (
            master["sample_targets"][family]
        )

        source_totals.update(sources)

    assert sum(source_totals.values()) == 2300

    expected_sources = {
        "custom": master["source_targets"]["custom_adversarial"],
        "bipia": master["source_targets"]["bipia_derived"],
        "tensor_trust": master["source_targets"]["tensor_trust_derived"],
        "deepset": master["source_targets"]["deepset_derived"]
    }

    assert dict(source_totals) == expected_sources

    assignments = partition_documents(
        qa_records,
        rng
    )

    assert len(assignments) == 100

    pools = defaultdict(list)

    for qa in qa_records:

        partition = assignments[
            qa["document_id"]
        ]

        pools[partition].append(qa)

    plans = []

    pair_counter = 0

    def add_slot(
        qa,
        family,
        source,
        split,
        pair_id=None
    ):

        plans.append({
            "planned_sample_id": (
                f"V2_ADV_{len(plans) + 1:05d}"
            ),

            "qa_id": qa["qa_id"],

            "document_id": qa["document_id"],

            "domain": qa["domain"],

            "attack_family": family,

            "planned_source_group": source,

            "split": split,

            "pair_id": pair_id,

            "template_partition": (
                "held_out"
                if split == "test_unseen"
                else "development"
            ),

            "attack_subcategory": None,

            "attack_objective": None,

            "attack_template_id": None,

            "success_criteria": None,

            "materialization_status": "pending",

            "source_review_status": "pending",

            "evaluation_ready": False
        })

    for family, sources in allocation.items():

        for source, count in sources.items():

            if count == 0:
                continue

            assert count % 5 == 0

            train_count = count * 40 // 100

            val_count = count * 20 // 100

            test_count = count * 20 // 100

            train_anchors = select_anchors(
                pools["train"],
                train_count,
                rng
            )

            val_anchors = select_anchors(
                pools["validation"],
                val_count,
                rng
            )

            test_anchors = select_anchors(
                pools["test"],
                test_count,
                rng
            )

            for qa in train_anchors:

                add_slot(
                    qa, family, source, "train"
                )

            for qa in val_anchors:

                add_slot(
                    qa, family, source, "validation"
                )

            for qa in test_anchors:

                pair_counter += 1

                pair_id = f"V2_PAIR_{pair_counter:04d}"

                add_slot(
                    qa,
                    family,
                    source,
                    "test_seen",
                    pair_id
                )

                add_slot(
                    qa,
                    family,
                    source,
                    "test_unseen",
                    pair_id
                )

    assert len(plans) == 2300

    assert len({
        p["planned_sample_id"] for p in plans
    }) == 2300

    split_counts = Counter(
        p["split"] for p in plans
    )

    expected_splits = {
        "train": 920,
        "validation": 460,
        "test_seen": 460,
        "test_unseen": 460
    }

    assert dict(split_counts) == expected_splits

    pairs = defaultdict(list)

    for plan in plans:

        if plan["pair_id"]:

            pairs[plan["pair_id"]].append(plan)

    assert len(pairs) == 460

    for pair_id, records in pairs.items():

        assert len(records) == 2

        assert {
            r["split"] for r in records
        } == {"test_seen", "test_unseen"}

        for field in [
            "qa_id",
            "document_id",
            "attack_family",
            "planned_source_group"
        ]:

            assert records[0][field] == records[1][field]

        assert (
            records[0]["template_partition"]
            != records[1]["template_partition"]
        )

    for plan in plans:

        expected_partition = (
            "test"
            if plan["split"].startswith("test")
            else plan["split"]
        )

        assert assignments[
            plan["document_id"]
        ] == expected_partition

    # Inspect benign compatibility with the
    # proposed document-level partition.

    benign_partition_counts = Counter(
        assignments[r["document_id"]]
        for r in benign_records
    )

    benign_category_by_partition = defaultdict(Counter)

    for record in benign_records:

        partition = assignments[
            record["document_id"]
        ]

        benign_category_by_partition[partition][
            record["benign_category"]
        ] += 1

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    save_jsonl(
        PROCESSED / "adversarial_sample_plan.jsonl",
        plans
    )

    save_json(
        METADATA / "document_assignments.json",
        assignments
    )

    report = {
        "dataset_version": "2.0-dev",

        "planned_adversarial_samples": len(plans),

        "family_distribution": dict(
            Counter(
                p["attack_family"] for p in plans
            )
        ),

        "source_distribution": dict(source_totals),

        "split_distribution": dict(split_counts),

        "document_partitions": dict(
            Counter(assignments.values())
        ),

        "qa_partitions": {
            name: len(records)
            for name, records in pools.items()
        },

        "matched_test_pairs": len(pairs),

        "planned_source_pair_mismatches": 0,

        "current_benign_partition_counts": dict(
            benign_partition_counts
        ),

        "benign_category_by_partition": {
            name: dict(counts)
            for name, counts
            in benign_category_by_partition.items()
        },

        "target_benign_partition_counts": {
            "train": 280,
            "validation": 140,
            "test": 280
        },

        "benign_rebalancing_required": (
            dict(benign_partition_counts)
            != {
                "train": 280,
                "validation": 140,
                "test": 280
            }
        ),

        "allocation_validation_passed": True,

        "external_source_feasibility_verified": False,

        "actual_template_separation_verified": False,

        "adversarial_samples_materialized": 0,

        "research_validation_complete": False
    }

    save_json(
        METADATA / "adversarial_plan_statistics.json",
        report
    )

    print("=" * 60)
    print("ADVERSARIAL BENCHMARK ALLOCATION")
    print("=" * 60)

    print(json.dumps(report, indent=2))

    print("\nADVERSARIAL PLANNING COMPLETED")


if __name__ == "__main__":
    main()
