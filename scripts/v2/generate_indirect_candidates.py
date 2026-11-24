
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
    / "indirect_scenarios_v2.json"
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

        if slot["attack_family"]
        == "indirect_injection"

        and slot["planned_source_group"]
        == "custom"
    ]

    assert len(slots) == 250

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
            f"IND_{partition.upper()}_"
            f"{scenario.upper()}"
        )

        sample = create_base_sample(
            slot,
            qa
        )

        sample.update({

            "attack_subcategory": scenario,

            "attack_channel": "retrieved_context",

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
                "retrieved_context_security_test"
            ),

            "document_format": "txt",

            "planned_document_format": (
                config["document_formats"][
                    index % len(
                        config["document_formats"]
                    )
                ]
            ),

            "evaluation_modes": (
                config["evaluation_modes"]
            ),

            "retrieval_metadata": {

                "target_document_id": (
                    qa["document_id"]
                ),

                "retrieved_document_ids": [],

                "target_document_retrieved": None,

                "target_chunk_retrieved": None,

                "retrieval_rank": None,

                "retrieval_score": None,

                "attack_exposed_to_model": None
            },

            "context_modified": False,

            "fixture_review_complete": False,

            "materialization_status": (
                "pending_fixture_review"
            ),

            "annotation_status": "pending",

            "evaluation_ready": False
        })

        samples.append(sample)

    # Validate sample allocation.

    assert len(samples) == 250

    assert len({
        r["sample_id"] for r in samples
    }) == 250

    assert all(
        r["user_input"] == r["question"]
        for r in samples
    )

    split_counts = Counter(
        r["split"] for r in samples
    )

    assert dict(split_counts) == {
        "train": 100,
        "validation": 50,
        "test_seen": 50,
        "test_unseen": 50
    }

    pairs = defaultdict(list)

    for sample in samples:

        if sample["pair_id"]:

            pairs[sample["pair_id"]].append(
                sample
            )

    assert len(pairs) == 50

    for records in pairs.values():

        assert len(records) == 2

        for field in [
            "qa_id",
            "document_id",
            "question",
            "attack_family",
            "source",
            "attack_objective",
            "success_criteria",
            "planned_document_format"
        ]:

            assert (
                records[0][field]
                == records[1][field]
            )

    save_jsonl(
        PROCESSED
        / "custom_indirect_candidates.jsonl",
        samples
    )

    report = {

        "total_candidates": len(samples),

        "attack_family": "indirect_injection",

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

        "scenario_label_overlap": 0,

        "legitimate_questions_preserved": True,

        "allocation_validation_passed": True,

        "materialized_samples": 0,

        "retrieval_exposure_verified": False,

        "manual_review_complete": False,

        "research_validation_complete": False
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        METADATA
        / "custom_indirect_statistics.json"
    )

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("=" * 60)

    print("CUSTOM INDIRECT INJECTION SCENARIOS")

    print("=" * 60)

    print(json.dumps(report, indent=2))

    print("\nINDIRECT PLANNING COMPLETED")


if __name__ == "__main__":

    main()
