
import json

from collections import Counter

from scenario_utils import (
    PROCESSED,
    METADATA,
    load_jsonl,
    save_jsonl,
    create_base_sample
)


def main():

    plan = load_jsonl(
        PROCESSED
        / "adversarial_sample_plan_revised.jsonl"
    )

    existing = load_jsonl(
        PROCESSED
        / "custom_benchmark_draft.jsonl"
    )

    qa_pairs = load_jsonl(
        PROCESSED
        / "expanded_qa_pairs.jsonl"
    )

    assert len(plan) == 2300
    assert len(existing) == 2000
    assert len(qa_pairs) == 500

    qa_lookup = {
        r["qa_id"]: r
        for r in qa_pairs
    }

    existing_lookup = {
        r["sample_id"]: r
        for r in existing
        if r["label"] == "adversarial"
    }

    benign = [
        r for r in existing
        if r["label"] == "benign"
    ]

    custom_slots = [
        r for r in plan
        if r["planned_source_group"] == "custom"
    ]

    assert len(benign) == 700
    assert len(custom_slots) == 1450

    custom_ids = {
        r["planned_sample_id"]
        for r in custom_slots
    }

    retained = []
    additional = []
    superseded = []

    for sample_id, sample in existing_lookup.items():

        if sample_id not in custom_ids:

            superseded.append(sample)

    for slot in custom_slots:

        sample_id = slot["planned_sample_id"]

        if sample_id in existing_lookup:

            sample = existing_lookup[sample_id]

            for field in [
                "qa_id",
                "document_id",
                "attack_family",
                "split",
                "pair_id",
                "template_partition"
            ]:

                assert sample[field] == slot[field]

            retained.append(sample)

        else:

            qa = qa_lookup[slot["qa_id"]]

            sample = create_base_sample(
                slot,
                qa
            )

            sample.update({

                "source": "custom_adversarial",

                "allocation_revision": 2,

                "materialization_status":
                    "pending_scenario_assignment",

                "annotation_status": "pending",

                "evaluation_ready": False
            })

            additional.append(sample)

    adversarial = retained + additional

    combined = benign + adversarial

    assert len(retained) == 1150
    assert len(additional) == 300
    assert len(superseded) == 150

    assert len(adversarial) == 1450
    assert len(combined) == 2150

    assert len({
        r["sample_id"]
        for r in combined
    }) == 2150

    assert {
        r["sample_id"]
        for r in adversarial
    } == custom_ids

    expected_families = {

        "direct_injection": 200,

        "instruction_manipulation": 250,

        "obfuscation": 500,

        "indirect_injection": 100,

        "multi_stage": 400
    }

    family_counts = Counter(

        r["attack_family"]

        for r in adversarial
    )

    assert dict(family_counts) == expected_families

    save_jsonl(

        PROCESSED
        / "custom_benchmark_revised.jsonl",

        combined
    )

    save_jsonl(

        PROCESSED
        / "additional_custom_candidates.jsonl",

        additional
    )

    save_jsonl(

        PROCESSED
        / "superseded_custom_candidates.jsonl",

        superseded
    )

    report = {

        "dataset_version": "2.0-dev",

        "allocation_revision": 2,

        "benign_records": len(benign),

        "retained_adversarial_records":
            len(retained),

        "additional_adversarial_candidates":
            len(additional),

        "superseded_adversarial_candidates":
            len(superseded),

        "total_custom_adversarial":
            len(adversarial),

        "total_custom_records":
            len(combined),

        "family_distribution":
            dict(family_counts),

        "additional_family_distribution": dict(

            Counter(
                r["attack_family"]
                for r in additional
            )
        ),

        "materialized_adversarial_records": sum(

            r.get("materialization_status")
            == "complete"

            for r in adversarial
        ),

        "original_files_preserved": True,

        "allocation_reconciliation_passed": True,

        "fixture_completion_verified": False,

        "research_validation_complete": False
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        METADATA
        / "custom_reconciliation_report.json"
    )

    output.write_text(

        json.dumps(report, indent=2),

        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    print("\nCUSTOM BENCHMARK RECONCILIATION COMPLETED")


if __name__ == "__main__":
    main()
