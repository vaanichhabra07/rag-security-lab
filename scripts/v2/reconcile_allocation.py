
import json

from collections import Counter, defaultdict
from copy import deepcopy

from scenario_utils import (
    ROOT,
    PROCESSED,
    METADATA,
    load_jsonl,
    save_jsonl
)


CONFIG = (
    ROOT / "configs"
    / "adversarial_allocation_v2_revised.json"
)


def main():

    config = json.loads(
        CONFIG.read_text(encoding="utf-8")
    )

    original = load_jsonl(
        PROCESSED / "adversarial_sample_plan.jsonl"
    )

    records = deepcopy(original)

    assert len(records) == 2300

    allocation = config["family_source_allocation"]

    percentages = config["split_percentages"]

    for family, sources in allocation.items():

        # Process train, validation and matched test
        # groups independently.

        for split in ["train", "validation", "test_seen"]:

            groups = defaultdict(list)

            for record in records:

                if record["attack_family"] != family:
                    continue

                if split == "test_seen":

                    if record["split"] in [
                        "test_seen",
                        "test_unseen"
                    ]:

                        groups[record["pair_id"]].append(
                            record
                        )

                elif record["split"] == split:

                    groups[
                        record["planned_sample_id"]
                    ].append(record)

            if split == "test_seen":

                assert all(
                    len(group) == 2
                    for group in groups.values()
                )

            targets = {}

            for source, count in sources.items():

                assert (
                    count * percentages[split]
                ) % 100 == 0

                targets[source] = (
                    count * percentages[split]
                ) // 100

            assert sum(targets.values()) == len(groups)

            retained = Counter()

            unassigned = []

            # Preserve original source where possible.

            for key in sorted(groups):

                group = groups[key]

                old_source = group[0][
                    "planned_source_group"
                ]

                assert all(
                    r["planned_source_group"] == old_source
                    for r in group
                )

                if retained[old_source] < targets.get(
                    old_source, 0
                ):

                    retained[old_source] += 1

                else:

                    unassigned.append(group)

            for source, target in targets.items():

                deficit = target - retained[source]

                for _ in range(deficit):

                    group = unassigned.pop(0)

                    for record in group:

                        record["planned_source_group"] = (
                            source
                        )

                    retained[source] += 1

            assert not unassigned

            assert all(
                retained[source] == target
                for source, target in targets.items()
            )

    original_lookup = {

        r["planned_sample_id"]: r

        for r in original
    }

    changed = []

    for record in records:

        old = original_lookup[
            record["planned_sample_id"]
        ]

        for field in [
            "qa_id",
            "document_id",
            "domain",
            "attack_family",
            "split",
            "pair_id",
            "template_partition"
        ]:

            assert record[field] == old[field]

        if (
            record["planned_source_group"]
            != old["planned_source_group"]
        ):

            changed.append({

                "sample_id": record[
                    "planned_sample_id"
                ],

                "attack_family": record[
                    "attack_family"
                ],

                "split": record["split"],

                "previous_source": old[
                    "planned_source_group"
                ],

                "revised_source": record[
                    "planned_source_group"
                ]
            })

    source_counts = Counter(
        r["planned_source_group"]
        for r in records
    )

    assert dict(source_counts) == {
        "custom": 1450,
        "bipia": 500,
        "tensor_trust": 350
    }

    family_source_counts = Counter(
        (
            r["attack_family"],
            r["planned_source_group"]
        )
        for r in records
    )

    for family, sources in allocation.items():

        for source, expected in sources.items():

            assert family_source_counts[
                (family, source)
            ] == expected

    split_counts = Counter(
        r["split"] for r in records
    )

    assert dict(split_counts) == {
        "train": 920,
        "validation": 460,
        "test_seen": 460,
        "test_unseen": 460
    }

    pairs = defaultdict(list)

    for record in records:

        if record.get("pair_id"):

            pairs[record["pair_id"]].append(record)

    assert len(pairs) == 460

    for group in pairs.values():

        assert len(group) == 2

        assert {
            r["split"] for r in group
        } == {
            "test_seen",
            "test_unseen"
        }

        for field in [
            "qa_id",
            "document_id",
            "attack_family",
            "planned_source_group"
        ]:

            assert group[0][field] == group[1][field]

    save_jsonl(
        PROCESSED / "adversarial_sample_plan_revised.jsonl",
        records
    )

    save_jsonl(
        PROCESSED / "allocation_changes.jsonl",
        changed
    )

    report = {

        "total_adversarial": len(records),

        "source_distribution": dict(source_counts),

        "split_distribution": dict(split_counts),

        "source_assignments_changed": len(changed),

        "matched_test_pairs": len(pairs),

        "source_pair_mismatches": 0,

        "original_sample_ids_preserved": True,

        "document_assignments_preserved": True,

        "original_plan_modified": False,

        "existing_custom_draft_reconciled": False,

        "allocation_validation_passed": True,

        "research_validation_complete": False
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        METADATA / "allocation_revision_report.json"
    )

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    print("\nALLOCATION RECONCILIATION COMPLETED")


if __name__ == "__main__":
    main()
