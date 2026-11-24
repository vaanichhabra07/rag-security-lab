
import json

from collections import Counter, defaultdict

from scenario_utils import (
    ROOT,
    PROCESSED,
    METADATA,
    load_jsonl,
    save_jsonl,
    text_hash,
    create_base_sample
)


CONFIG = (
    ROOT / "configs"
    / "instruction_scenarios_v2.json"
)


def main():

    config = json.loads(
        CONFIG.read_text(encoding="utf-8")
    )

    plans = load_jsonl(
        PROCESSED / "adversarial_sample_plan.jsonl"
    )

    qa_records = load_jsonl(
        PROCESSED / "expanded_qa_pairs.jsonl"
    )

    qa_lookup = {
        qa["qa_id"]: qa
        for qa in qa_records
    }

    slots = [
        slot for slot in plans

        if slot["attack_family"]
        == "instruction_manipulation"

        and slot["planned_source_group"]
        == "custom"
    ]

    assert len(slots) == 200

    counters = defaultdict(int)

    samples = []

    for slot in slots:

        qa = qa_lookup[slot["qa_id"]]

        split = slot["split"]

        partition = slot["template_partition"]

        index = counters[split]

        counters[split] += 1

        scenarios = (
            config["held_out_scenarios"]
            if partition == "held_out"
            else config["development_scenarios"]
        )

        scenario = scenarios[
            index % len(scenarios)
        ]

        targets = config["synthetic_targets"]

        target = targets[
            (index // len(scenarios))
            % len(targets)
        ]

        objectives = config["security_objectives"]

        objective = objectives[
            index % len(objectives)
        ]

        sample = create_base_sample(
            slot,
            qa
        )

        if objective == "output_manipulation":

            criteria = {
                "type": "marker_present",
                "target": target,
                "case_sensitive": True
            }

        elif objective == "answer_integrity_violation":

            criteria = {
                "type": "semantic_review",
                "goal": "incorrect_institutional_answer",
                "requires_ground_truth": True
            }

        else:

            criteria = {
                "type": "semantic_review",
                "goal": "unjustified_task_denial",
                "requires_ground_truth": True
            }

        sample.update({

            "attack_subcategory": (
                scenario["subcategory"]
            ),

            "attack_channel": "user_input",

            "attack_template_id": (
                scenario["scenario_id"]
            ),

            "attack_objective": objective,

            "attack_goal": objective,

            "success_criteria": criteria,

            "trusted_policy_id": (
                config["trusted_policy"]["policy_id"]
            ),

            "source": "custom_adversarial",

            "source_id": scenario["scenario_id"],

            "scenario_description": (
                scenario["subcategory"]
            ),

            "synthetic_target": target,

            "fixture_type": (
                "instruction_hierarchy_test"
            ),

            "template_hash": text_hash(
                json.dumps(
                    scenario,
                    sort_keys=True
                )
            ),

            "materialization_status": (
                "pending_fixture_review"
            ),

            "annotation_status": "pending",

            "evaluation_ready": False
        })

        samples.append(sample)

    # Validate record allocation

    assert len(samples) == 200

    assert len({
        r["sample_id"] for r in samples
    }) == 200

    split_counts = Counter(
        r["split"] for r in samples
    )

    assert dict(split_counts) == {
        "train": 80,
        "validation": 40,
        "test_seen": 40,
        "test_unseen": 40
    }

    development_ids = {
        r["attack_template_id"]
        for r in samples

        if r["split"] in {
            "train",
            "validation"
        }
    }

    unseen_ids = {
        r["attack_template_id"]
        for r in samples

        if r["split"] == "test_unseen"
    }

    assert not (
        development_ids & unseen_ids
    )

    pairs = defaultdict(list)

    for sample in samples:

        if sample["pair_id"]:

            pairs[sample["pair_id"]].append(
                sample
            )

    assert len(pairs) == 40

    for records in pairs.values():

        assert len(records) == 2

        for field in [
            "qa_id",
            "document_id",
            "attack_family",
            "source",
            "attack_objective",
            "success_criteria"
        ]:

            assert (
                records[0][field]
                == records[1][field]
            )

    save_jsonl(
        PROCESSED
        / "custom_instruction_candidates.jsonl",
        samples
    )

    report = {

        "total_candidates": len(samples),

        "attack_family": (
            "instruction_manipulation"
        ),

        "split_distribution": dict(
            split_counts
        ),

        "subcategory_distribution": dict(
            Counter(
                r["attack_subcategory"]
                for r in samples
            )
        ),

        "objective_distribution": dict(
            Counter(
                r["attack_objective"]
                for r in samples
            )
        ),

        "unique_scenario_ids": len({
            r["attack_template_id"]
            for r in samples
        }),

        "matched_test_pairs": len(pairs),

        "scenario_id_overlap": 0,

        "allocation_validation_passed": True,

        "materialized_samples": 0,

        "manual_review_complete": False,

        "research_validation_complete": False
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        METADATA
        / "custom_instruction_statistics.json"
    )

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("=" * 60)

    print("INSTRUCTION MANIPULATION CATALOGUE")

    print("=" * 60)

    print(json.dumps(report, indent=2))

    print("\nSCENARIO ALLOCATION COMPLETED")


if __name__ == "__main__":
    main()
