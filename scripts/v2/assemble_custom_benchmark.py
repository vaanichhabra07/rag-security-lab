
import json

from collections import Counter, defaultdict

from scenario_utils import (
    PROCESSED,
    METADATA,
    load_jsonl,
    save_jsonl
)


INPUT_FILES = {

    "benign_samples.jsonl": 700,

    "custom_direct_samples.jsonl": 200,

    "custom_instruction_candidates.jsonl": 200,

    "custom_obfuscation_candidates.jsonl": 300,

    "custom_indirect_candidates.jsonl": 250,

    "custom_multistage_candidates.jsonl": 350
}


EXPECTED_FAMILIES = {

    "none": 700,

    "direct_injection": 200,

    "instruction_manipulation": 200,

    "obfuscation": 300,

    "indirect_injection": 250,

    "multi_stage": 350
}


def main():

    records = []

    file_counts = {}

    for filename, expected_count in INPUT_FILES.items():

        path = PROCESSED / filename

        if not path.exists():

            raise FileNotFoundError(path)

        samples = load_jsonl(path)

        assert len(samples) == expected_count, (
            f"Incorrect count in {filename}"
        )

        file_counts[filename] = len(samples)

        records.extend(samples)

    assert len(records) == 2000

    ids = [
        r["sample_id"]
        for r in records
    ]

    assert len(set(ids)) == len(ids), (
        "Duplicate sample IDs detected"
    )

    benign = [
        r for r in records
        if r["label"] == "benign"
    ]

    adversarial = [
        r for r in records
        if r["label"] == "adversarial"
    ]

    assert len(benign) == 700

    assert len(adversarial) == 1300

    family_counts = Counter(
        r["attack_family"]
        for r in records
    )

    assert dict(family_counts) == EXPECTED_FAMILIES

    # Verify alignment with the original allocation.

    plan = load_jsonl(
        PROCESSED / "adversarial_sample_plan.jsonl"
    )

    custom_plan = {

        r["planned_sample_id"]: r

        for r in plan

        if r["planned_source_group"] == "custom"
    }

    assert len(custom_plan) == 1300

    assert {
        r["sample_id"]
        for r in adversarial
    } == set(custom_plan)

    for sample in adversarial:

        slot = custom_plan[
            sample["sample_id"]
        ]

        for field in [
            "qa_id",
            "document_id",
            "attack_family",
            "split",
            "pair_id",
            "template_partition"
        ]:

            assert sample[field] == slot[field], (
                f"Plan mismatch: "
                f"{sample['sample_id']} - {field}"
            )

    # Separate materialized records from candidates.

    materialized = []

    pending = []

    for sample in records:

        if sample["label"] == "benign":

            materialized.append(sample)

        elif (
            sample.get("materialization_status")
            == "complete"
        ):

            materialized.append(sample)

        else:

            pending.append(sample)

    assert len(materialized) == 900

    assert len(pending) == 1100

    # Verify matched seen/unseen test pairs.

    pairs = defaultdict(list)

    for sample in adversarial:

        if sample.get("pair_id"):

            pairs[sample["pair_id"]].append(
                sample
            )

    assert len(pairs) == 260

    for pair_id, group in pairs.items():

        assert len(group) == 2, pair_id

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
            "source",
            "attack_objective",
            "success_criteria"
        ]:

            assert (
                group[0].get(field)
                == group[1].get(field)
            ), f"{pair_id}: {field}"

    # Verify template ID separation.

    development_templates = {

        r["attack_template_id"]

        for r in adversarial

        if r["split"] in {
            "train",
            "validation"
        }
    }

    unseen_templates = {

        r["attack_template_id"]

        for r in adversarial

        if r["split"] == "test_unseen"
    }

    overlap = (
        development_templates
        & unseen_templates
    )

    assert not overlap

    save_jsonl(
        PROCESSED / "custom_benchmark_draft.jsonl",
        records
    )

    save_jsonl(
        PROCESSED / "custom_materialized.jsonl",
        materialized
    )

    save_jsonl(
        PROCESSED / "custom_pending_review.jsonl",
        pending
    )

    report = {

        "dataset_version": "2.0-dev",

        "total_custom_records": len(records),

        "benign_records": len(benign),

        "adversarial_records": len(adversarial),

        "materialized_records": len(materialized),

        "pending_candidates": len(pending),

        "remaining_external_allocation": 1000,

        "file_counts": file_counts,

        "family_distribution": dict(
            family_counts
        ),

        "adversarial_split_distribution": dict(
            Counter(
                r["split"]
                for r in adversarial
            )
        ),

        "planned_objective_distribution": dict(
            Counter(
                r["attack_objective"]
                for r in adversarial
            )
        ),

        "matched_test_pairs": len(pairs),

        "template_id_overlap": len(overlap),

        "structural_validation_passed": True,

        "fixture_validation_complete": False,

        "research_validation_complete": False,

        "ready_for_final_evaluation": False
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        METADATA
        / "custom_benchmark_statistics.json"
    )

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    print("\nCUSTOM BENCHMARK ASSEMBLY COMPLETED")


if __name__ == "__main__":

    main()
