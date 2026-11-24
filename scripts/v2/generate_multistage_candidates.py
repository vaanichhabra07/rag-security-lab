
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
    / "multistage_scenarios_v2.json"
)


def build_criteria(objective, target):

    if objective == "output_manipulation":

        return {
            "type": "marker_present",
            "target": target,
            "case_sensitive": True
        }

    if objective == "synthetic_information_disclosure":

        return {
            "type": "synthetic_disclosure_review",
            "requires_synthetic_secret": True,
            "requires_policy_verification": True
        }

    return {
        "type": "semantic_review",
        "goal": objective,
        "requires_ground_truth": True
    }


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

        if slot["attack_family"] == "multi_stage"

        and slot["planned_source_group"] == "custom"
    ]

    assert len(slots) == 350

    development = config["development_scenarios"]

    held_out = config["held_out_scenarios"]

    assert not (
        set(development) & set(held_out)
    )

    counters = defaultdict(int)

    samples = []

    for slot in slots:

        qa = qa_lookup[slot["qa_id"]]

        split = slot["split"]

        partition = slot["template_partition"]

        index = counters[split]

        counters[split] += 1

        scenarios = (
            held_out
            if partition == "held_out"
            else development
        )

        scenario = scenarios[
            index % len(scenarios)
        ]

        objectives = config["security_objectives"]

        objective = objectives[
            index % len(objectives)
        ]

        targets = config["synthetic_targets"]

        target = targets[
            (index // len(objectives))
            % len(targets)
        ]

        scenario_id = (
            f"MULTI_{partition.upper()}_"
            f"{scenario.upper()}"
        )

        planned_turns = 2 + (index % 2)

        sample = create_base_sample(
            slot,
            qa
        )

        sample.update({

            "attack_subcategory": scenario,

            "attack_channel": (
                "conversation_history"
            ),

            "attack_template_id": scenario_id,

            "attack_objective": objective,

            "attack_goal": objective,

            "success_criteria": build_criteria(
                objective,
                target
            ),

            "source": "custom_adversarial",

            "source_id": scenario_id,

            "synthetic_target": target,

            "template_hash": text_hash(
                scenario_id
            ),

            "fixture_type": (
                "multi_stage_security_test"
            ),

            "conversation_mode": (
                "live_model_history"
            ),

            "planned_conversation_turns": (
                planned_turns
            ),

            "conversation_plan": [],

            "conversation_history": [],

            "requires_live_execution": True,

            "requires_session_isolation": True,

            "multistage_dependency_verified": False,

            "fixture_review_complete": False,

            "materialization_status": (
                "pending_fixture_review"
            ),

            "annotation_status": "pending",

            "evaluation_ready": False
        })

        samples.append(sample)

    # Structural validation

    assert len(samples) == 350

    assert len({
        r["sample_id"] for r in samples
    }) == 350

    split_counts = Counter(
        r["split"] for r in samples
    )

    assert dict(split_counts) == {
        "train": 140,
        "validation": 70,
        "test_seen": 70,
        "test_unseen": 70
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

    assert len(pairs) == 70

    for records in pairs.values():

        assert len(records) == 2

        assert {
            r["split"] for r in records
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
            "success_criteria",
            "planned_conversation_turns"
        ]:

            assert (
                records[0][field]
                == records[1][field]
            )

    save_jsonl(
        PROCESSED
        / "custom_multistage_candidates.jsonl",
        samples
    )

    report = {

        "total_candidates": len(samples),

        "attack_family": "multi_stage",

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

        "planned_turn_distribution": dict(
            Counter(
                r["planned_conversation_turns"]
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

        "live_conversations_executed": 0,

        "multistage_dependency_verified": False,

        "manual_review_complete": False,

        "research_validation_complete": False
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        METADATA
        / "custom_multistage_statistics.json"
    )

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("=" * 60)

    print("CUSTOM MULTI-STAGE SCENARIOS")

    print("=" * 60)

    print(json.dumps(report, indent=2))

    print("\nMULTI-STAGE PLANNING COMPLETED")


if __name__ == "__main__":

    main()
