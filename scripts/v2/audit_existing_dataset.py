
import json
import csv
import hashlib
import re

from pathlib import Path
from collections import Counter, defaultdict


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "processed"
    / "master_dataset.jsonl"
)

OUTPUT_DIR = (
    ROOT / "data" / "v2" / "metadata"
)

EXPECTED_FAMILIES = [
    "direct_injection",
    "instruction_manipulation",
    "obfuscation",
    "indirect_injection",
    "multi_stage"
]


def load_jsonl(path):

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def normalize(text):

    return re.sub(
        r"\s+",
        " ",
        str(text).casefold()
    ).strip()


def source_group(record):

    source = str(
        record.get("source", "")
    ).lower()

    if "bipia" in source:
        return "bipia"

    if "custom" in source:
        return "custom"

    return source or "unknown"


def get_payload(record):

    question = record.get("question") or ""

    user_input = record.get("user_input") or ""

    clean = record.get("clean_context") or ""

    modified = record.get("modified_context") or ""

    user_delta = ""

    context_delta = ""

    if user_input != question:

        if user_input.startswith(question):
            user_delta = user_input[len(question):]
        else:
            user_delta = user_input

    if modified != clean:

        if modified.startswith(clean):
            context_delta = modified[len(clean):]
        else:
            context_delta = modified

    history = record.get(
        "conversation_history"
    ) or []

    return {
        "user_delta": normalize(user_delta),
        "context_delta": normalize(context_delta),
        "history": history
    }


def payload_hash(record):

    payload = get_payload(record)

    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def audit_sample(record):

    issues = []

    label = record.get("label")

    family = record.get("attack_family")

    source = source_group(record)

    criteria = record.get(
        "success_criteria"
    ) or {}

    if not record.get("expected_answer"):
        issues.append("missing_expected_answer")

    if label == "benign":

        if get_payload(record) != {
            "user_delta": "",
            "context_delta": "",
            "history": []
        }:
            issues.append(
                "unexpected_benign_modification"
            )

        return issues

    if not record.get("attack_template_id"):
        issues.append("missing_template_id")

    if not record.get("attack_goal"):
        issues.append("missing_attack_goal")

    if not criteria:
        issues.append("missing_success_criteria")

    payload = get_payload(record)

    if not any(payload.values()):
        issues.append("missing_attack_payload")

    if source == "custom":

        issues.append(
            "custom_template_diversity_review"
        )

    if family == "obfuscation":

        issues.append(
            "verify_genuine_input_obfuscation"
        )

        if source == "bipia":
            issues.append(
                "review_output_transformation_label"
            )

    if family == "instruction_manipulation":

        issues.append(
            "verify_instruction_hierarchy_conflict"
        )

    if family == "multi_stage":

        issues.append(
            "requires_live_multiturn_evaluation"
        )

        history = record.get(
            "conversation_history"
        ) or []

        if not history:
            issues.append(
                "missing_conversation_history"
            )

    if criteria.get("type") == "semantic_review":

        issues.append(
            "semantic_rubric_validation_required"
        )

    if criteria.get("type") == "exact_match":

        issues.append(
            "partial_compliance_scoring_review"
        )

    if record.get("evaluation_ready") is False:

        issues.append(
            "evaluation_readiness_pending"
        )

    return issues


def recommendation(record, issues):

    critical = {
        "missing_expected_answer",
        "missing_template_id",
        "missing_attack_goal",
        "missing_success_criteria",
        "missing_attack_payload",
        "missing_conversation_history",
        "unexpected_benign_modification"
    }

    if critical.intersection(issues):
        return "repair_required"

    if record.get("label") == "benign":
        return "retain_candidate"

    family = record.get("attack_family")

    if family in {
        "obfuscation",
        "instruction_manipulation",
        "multi_stage"
    }:
        return "refinement_candidate"

    if source_group(record) == "bipia":
        return "external_template_review"

    return "retain_after_review"


def main():

    records = load_jsonl(INPUT_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    benign = [
        r for r in records
        if r.get("label") == "benign"
    ]

    adversarial = [
        r for r in records
        if r.get("label") == "adversarial"
    ]

    ids = [
        r.get("sample_id")
        for r in records
    ]

    family_counts = Counter(
        r.get("attack_family")
        for r in adversarial
    )

    source_counts = Counter(
        r.get("source", "unknown")
        for r in records
    )

    split_counts = Counter(
        r.get("split", "unknown")
        for r in records
    )

    objective_counts = Counter(
        str(r.get("attack_goal"))
        for r in adversarial
    )

    criteria_counts = Counter(
        (r.get("success_criteria") or {})
        .get("type", "missing")
        for r in adversarial
    )

    channel_counts = Counter(
        r.get("attack_channel", "unknown")
        for r in adversarial
    )

    audit_rows = []

    issue_counts = Counter()

    recommendation_counts = Counter()

    family_statistics = {}

    for family in EXPECTED_FAMILIES:

        subset = [
            r for r in adversarial
            if r.get("attack_family") == family
        ]

        templates = {
            r.get("attack_template_id")
            for r in subset
            if r.get("attack_template_id")
        }

        payloads = {
            payload_hash(r)
            for r in subset
        }

        family_statistics[family] = {
            "samples": len(subset),
            "unique_template_ids": len(templates),
            "unique_normalized_payloads": len(payloads),
            "payload_diversity_ratio": round(
                len(payloads) / len(subset), 4
            ) if subset else 0,
            "sources": dict(
                Counter(
                    source_group(r)
                    for r in subset
                )
            ),
            "channels": dict(
                Counter(
                    r.get("attack_channel")
                    for r in subset
                )
            )
        }

    for record in records:

        issues = audit_sample(record)

        decision = recommendation(
            record,
            issues
        )

        issue_counts.update(issues)

        recommendation_counts[decision] += 1

        audit_rows.append({
            "sample_id": record.get("sample_id"),
            "label": record.get("label"),
            "attack_family": record.get(
                "attack_family"
            ),
            "source": record.get("source"),
            "split": record.get("split"),
            "template_id": record.get(
                "attack_template_id"
            ),
            "issue_count": len(issues),
            "issues": "; ".join(issues),
            "recommendation": decision,
            "manual_review_status": "pending"
        })

    development = [
        r for r in adversarial
        if r.get("split") in {
            "train", "validation"
        }
    ]

    seen = [
        r for r in adversarial
        if r.get("split") == "test_seen"
    ]

    unseen = [
        r for r in adversarial
        if r.get("split") == "test_unseen"
    ]

    development_templates = {
        r["attack_template_id"]
        for r in development
        if r.get("attack_template_id")
    }

    unseen_templates = {
        r["attack_template_id"]
        for r in unseen
        if r.get("attack_template_id")
    }

    development_payloads = {
        payload_hash(r)
        for r in development
    }

    unseen_payloads = {
        payload_hash(r)
        for r in unseen
    }

    seen_pairs = defaultdict(Counter)

    unseen_pairs = defaultdict(Counter)

    for record in seen:

        key = (
            record.get("qa_id"),
            record.get("attack_family")
        )

        seen_pairs[key][source_group(record)] += 1

    for record in unseen:

        key = (
            record.get("qa_id"),
            record.get("attack_family")
        )

        unseen_pairs[key][source_group(record)] += 1

    common_pairs = (
        set(seen_pairs) & set(unseen_pairs)
    )

    source_mismatches = sum(
        seen_pairs[key] != unseen_pairs[key]
        for key in common_pairs
    )

    report = {
        "dataset_version": "1.0-rc1",
        "total_samples": len(records),
        "benign_samples": len(benign),
        "adversarial_samples": len(adversarial),

        "integrity": {
            "expected_total": len(records) == 1500,
            "unique_sample_ids": (
                len(set(ids)) == len(ids)
                and all(ids)
            ),
            "expected_benign": len(benign) == 250,
            "expected_adversarial": (
                len(adversarial) == 1250
            ),
            "expected_families": (
                set(family_counts)
                == set(EXPECTED_FAMILIES)
            )
        },

        "family_distribution": dict(family_counts),
        "source_distribution": dict(source_counts),
        "split_distribution": dict(split_counts),
        "attack_objectives": dict(objective_counts),
        "success_criteria": dict(criteria_counts),
        "attack_channels": dict(channel_counts),

        "family_statistics": family_statistics,

        "generalization_audit": {
            "development_template_count": len(
                development_templates
            ),
            "unseen_template_count": len(
                unseen_templates
            ),
            "template_id_overlap": len(
                development_templates
                & unseen_templates
            ),
            "normalized_payload_overlap": len(
                development_payloads
                & unseen_payloads
            ),
            "matched_qa_family_groups": len(
                common_pairs
            ),
            "unmatched_qa_family_groups": len(
                set(seen_pairs) ^ set(unseen_pairs)
            ),
            "source_pair_mismatches": (
                source_mismatches
            )
        },

        "quality_issues": dict(issue_counts),

        "recommendations": dict(
            recommendation_counts
        ),

        "audit_type": "automated_heuristic",
        "manual_review_complete": False,
        "research_validation_complete": False,
        "original_dataset_modified": False
    }

    report_path = (
        OUTPUT_DIR / "v1_quality_audit.json"
    )

    with report_path.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    csv_path = (
        OUTPUT_DIR / "v1_sample_review.csv"
    )

    if audit_rows:

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    audit_rows[0].keys()
                )
            )

            writer.writeheader()
            writer.writerows(audit_rows)

    print("=" * 60)
    print("EXISTING DATASET QUALITY AUDIT")
    print("=" * 60)

    print(f"Total samples: {len(records)}")
    print(f"Benign samples: {len(benign)}")
    print(f"Adversarial samples: {len(adversarial)}")

    print("\nFAMILY STATISTICS")

    for family, stats in family_statistics.items():

        print(f"\n{family}")

        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("\nQUALITY ISSUES")

    for issue, count in issue_counts.most_common():
        print(f"{issue}: {count}")

    print("\nRECOMMENDATIONS")

    for decision, count in (
        recommendation_counts.most_common()
    ):
        print(f"{decision}: {count}")

    print("\nGENERALIZATION AUDIT")

    for key, value in report[
        "generalization_audit"
    ].items():
        print(f"{key}: {value}")

    print(f"\nJSON report: {report_path}")
    print(f"CSV review: {csv_path}")

    print("\nOriginal dataset unchanged.")

    if not all(report["integrity"].values()):

        raise SystemExit(
            "\nDATASET INTEGRITY CHECK FAILED"
        )

    print("\nDATASET QUALITY AUDIT COMPLETED")


if __name__ == "__main__":
    main()
