
import csv
import json

from collections import Counter

from scenario_utils import (
    PROCESSED,
    METADATA,
    load_jsonl
)


def audit_sample(sample):

    issues = []

    family = sample["attack_family"]

    if sample["label"] == "benign":

        if (
            sample["clean_context"]
            != sample["modified_context"]
        ):

            issues.append(
                "unexpected_context_modification"
            )

        if (
            sample.get("answerable")
            and not sample.get("expected_answer")
        ):

            issues.append(
                "missing_expected_answer"
            )

        if not sample.get("answerable"):

            issues.append(
                "unanswerability_review_required"
            )

        if sample.get("benign_category") == "multilingual":

            issues.append(
                "language_quality_review_required"
            )

    else:

        changed = any([

            sample["user_input"]
            != sample["question"],

            sample["modified_context"]
            != sample["clean_context"],

            bool(
                sample.get("conversation_history")
            ),

            bool(
                sample.get("conversation_plan")
            )
        ])

        if not changed:

            issues.append(
                "missing_adversarial_fixture"
            )

        if not sample.get("success_criteria"):

            issues.append(
                "missing_success_criteria"
            )

        if family == "direct_injection":

            issues.append(
                "strategy_diversity_review_required"
            )

        if family == "instruction_manipulation":

            issues.append(
                "instruction_conflict_review_required"
            )

        if family == "obfuscation":

            if not sample.get(
                "semantic_equivalence_verified"
            ):

                issues.append(
                    "semantic_equivalence_unverified"
                )

            if not sample.get(
                "genuine_obfuscation_verified"
            ):

                issues.append(
                    "input_obfuscation_unverified"
                )

        if family == "indirect_injection":

            if (
                sample["modified_context"]
                == sample["clean_context"]
            ):

                issues.append(
                    "retrieved_context_fixture_missing"
                )

            issues.append(
                "retrieval_exposure_review_required"
            )

        if family == "multi_stage":

            if not sample.get("conversation_plan"):

                issues.append(
                    "conversation_plan_missing"
                )

            if not sample.get(
                "multistage_dependency_verified"
            ):

                issues.append(
                    "multistage_dependency_unverified"
                )

    if sample.get("annotation_status") != "approved":

        issues.append(
            "manual_annotation_pending"
        )

    return issues


def main():

    records = load_jsonl(
        PROCESSED / "custom_benchmark_draft.jsonl"
    )

    assert len(records) == 2000

    issue_counts = Counter()

    review_rows = []

    for sample in records:

        issues = audit_sample(sample)

        issue_counts.update(issues)

        review_rows.append({

            "sample_id": sample["sample_id"],

            "qa_id": sample["qa_id"],

            "document_id": sample["document_id"],

            "label": sample["label"],

            "attack_family": sample["attack_family"],

            "attack_subcategory": sample.get(
                "attack_subcategory"
            ),

            "attack_objective": sample.get(
                "attack_objective"
            ),

            "split": sample.get("split"),

            "materialization_status": sample.get(
                "materialization_status",
                "benign_record"
            ),

            "issue_count": len(issues),

            "quality_flags": " | ".join(issues),

            "review_status": "pending",

            "review_notes": ""
        })

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        METADATA / "custom_fixture_review.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(review_rows[0])
        )

        writer.writeheader()

        writer.writerows(review_rows)

    report = {

        "total_records_audited": len(records),

        "records_with_quality_flags": sum(
            r["issue_count"] > 0
            for r in review_rows
        ),

        "quality_flag_distribution": dict(
            issue_counts
        ),

        "automated_audit_complete": True,

        "manual_review_complete": False,

        "research_validation_complete": False
    }

    output = (
        METADATA / "custom_quality_audit.json"
    )

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    print("\nCUSTOM QUALITY AUDIT COMPLETED")


if __name__ == "__main__":

    main()
