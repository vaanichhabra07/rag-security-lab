
import json
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
PROCESSED = ROOT / "data" / "processed"
METADATA = ROOT / "data" / "metadata"

EXPECTED = {
    "train": 600,
    "validation": 300,
    "test_seen": 250,
    "test_unseen": 250,
    "test_benign": 100
}

results = []
warnings = []


def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def check(name, condition):
    condition = bool(condition)

    results.append({
        "check": name,
        "passed": condition
    })

    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")


def warn(message):
    warnings.append(message)
    print(f"[WARNING] {message}")


def values(records, field):
    return {
        r[field]
        for r in records
        if r.get(field) is not None
    }


def attack_fingerprint(record):

    question = record["question"]
    user_input = record["user_input"]

    clean = record["clean_context"]
    modified = record["modified_context"]

    user_delta = user_input.removeprefix(question)
    context_delta = modified.removeprefix(clean)

    payload = {
        "user_delta": user_delta.strip(),
        "context_delta": context_delta.strip(),
        "history": record["conversation_history"]
    }

    normalized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False
    ).casefold()

    normalized = " ".join(normalized.split())

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def main():

    datasets = {
        split: load_jsonl(
            SPLITS / f"{split}.jsonl"
        )
        for split in EXPECTED
    }

    master = load_jsonl(
        PROCESSED / "master_dataset.jsonl"
    )

    all_records = [
        r
        for records in datasets.values()
        for r in records
    ]

    print("\n=== 1. DATASET INTEGRITY ===")

    check("Master dataset has 1500 records",
          len(master) == 1500)

    check("Combined splits have 1500 records",
          len(all_records) == 1500)

    check("Unique sample IDs",
          len(values(all_records, "sample_id")) == 1500)

    master_lookup = {
        r["sample_id"]: r for r in master
    }

    split_lookup = {
        r["sample_id"]: r for r in all_records
    }

    check("Master and split records match",
          master_lookup == split_lookup)

    for split, expected in EXPECTED.items():

        check(
            f"{split}: {expected} records",
            len(datasets[split]) == expected
        )

        check(
            f"{split}: correct split labels",
            all(r["split"] == split
                for r in datasets[split])
        )

    print("\n=== 2. CLASS DISTRIBUTION ===")

    labels = Counter(
        r["label"] for r in all_records
    )

    check("250 benign samples",
          labels["benign"] == 250)

    check("1250 adversarial samples",
          labels["adversarial"] == 1250)

    families = [
        "direct_injection",
        "instruction_manipulation",
        "obfuscation",
        "indirect_injection",
        "multi_stage"
    ]

    family_counts = Counter(
        r["attack_family"]
        for r in all_records
    )

    for family in families:

        check(
            f"{family}: 250 samples",
            family_counts[family] == 250
        )

    print("\n=== 3. DOCUMENT LEAKAGE ===")

    train = datasets["train"]
    validation = datasets["validation"]

    seen = datasets["test_seen"]
    unseen = datasets["test_unseen"]

    test = (
        seen + unseen + datasets["test_benign"]
    )

    train_docs = values(train, "document_id")
    val_docs = values(validation, "document_id")
    test_docs = values(test, "document_id")

    check("Train-validation document overlap = 0",
          not train_docs & val_docs)

    check("Train-test document overlap = 0",
          not train_docs & test_docs)

    check("Validation-test document overlap = 0",
          not val_docs & test_docs)

    check("Train-test QA overlap = 0",
          not values(train, "qa_id")
          & values(test, "qa_id"))

    check("Validation-test QA overlap = 0",
          not values(validation, "qa_id")
          & values(test, "qa_id"))

    print("\n=== 4. TEMPLATE LEAKAGE ===")

    train_templates = values(
        train, "attack_template_id"
    )

    val_templates = values(
        validation, "attack_template_id"
    )

    seen_templates = values(
        seen, "attack_template_id"
    )

    unseen_templates = values(
        unseen, "attack_template_id"
    )

    check("Train-unseen template ID overlap = 0",
          not train_templates & unseen_templates)

    check("Validation-unseen template ID overlap = 0",
          not val_templates & unseen_templates)

    check("Seen-unseen template ID overlap = 0",
          not seen_templates & unseen_templates)

    check("Seen templates present in training",
          seen_templates.issubset(train_templates))

    print("\n=== 5. ATTACK CONTENT LEAKAGE ===")

    development_hashes = {
        attack_fingerprint(r)
        for r in train + validation
        if r["label"] == "adversarial"
    }

    unseen_hashes = {
        attack_fingerprint(r)
        for r in unseen
    }

    overlap = development_hashes & unseen_hashes

    check("Exact normalized payload overlap = 0",
          len(overlap) == 0)

    print(f"Overlapping payload hashes: {len(overlap)}")

    print("\n=== 6. PAIRED TEST VALIDATION ===")

    seen_pairs = Counter(
        (r["qa_id"], r["attack_family"])
        for r in seen
    )

    unseen_pairs = Counter(
        (r["qa_id"], r["attack_family"])
        for r in unseen
    )

    check("Seen and unseen QA-family pairs match",
          seen_pairs == unseen_pairs)

    seen_sources = {
        (r["qa_id"], r["attack_family"]):
        r["source"].startswith("BIPIA")
        for r in seen
    }

    unseen_sources = {
        (r["qa_id"], r["attack_family"]):
        r["source"].startswith("BIPIA")
        for r in unseen
    }

    source_mismatches = sum(
        seen_sources[key] != unseen_sources[key]
        for key in seen_sources.keys()
        & unseen_sources.keys()
    )

    if source_mismatches:
        warn(
            f"{source_mismatches} paired tests "
            "have different attack sources."
        )

    print("\n=== 7. SAMPLE QUALITY ===")

    benign = [
        r for r in all_records
        if r["label"] == "benign"
    ]

    attacks = [
        r for r in all_records
        if r["label"] == "adversarial"
    ]

    check("Benign samples are unmodified",
          all(
              r["user_input"] == r["question"]
              and r["clean_context"]
              == r["modified_context"]
              and not r["conversation_history"]
              for r in benign
          ))

    check("All adversarial samples contain changes",
          all(
              r["user_input"] != r["question"]
              or r["clean_context"]
              != r["modified_context"]
              or bool(r["conversation_history"])
              for r in attacks
          ))

    check("All samples have expected answers",
          all(r.get("expected_answer")
              for r in all_records))

    check("All attacks have success criteria",
          all(r.get("success_criteria")
              for r in attacks))

    check("All attacks have defined goals",
          all(r.get("attack_goal")
              for r in attacks))

    check("All attacks have template IDs",
          all(r.get("attack_template_id")
              for r in attacks))

    semantic_count = sum(
        r["success_criteria"].get("type")
        == "semantic_review"
        for r in attacks
    )

    if semantic_count:
        warn(
            f"{semantic_count} samples require "
            "validated semantic scoring."
        )

    warn(
        "Exact payload checks do not establish "
        "semantic independence of attack templates."
    )

    warn(
        "Custom attack diversity and family "
        "definitions require qualitative review."
    )

    print("\n=== FINAL VALIDATION SUMMARY ===")

    passed = sum(r["passed"] for r in results)
    failed = len(results) - passed

    report = {
        "total_samples": len(all_records),
        "checks_passed": passed,
        "checks_failed": failed,
        "checks": results,
        "warnings": warnings,
        "source_pair_mismatches": source_mismatches,
        "exact_payload_overlap": len(overlap),
        "semantic_review_samples": semantic_count,
        "structural_validation_passed": failed == 0,
        "research_validation_complete": False
    }

    METADATA.mkdir(parents=True, exist_ok=True)

    output = METADATA / "validation_report.json"

    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"Checks passed: {passed}")
    print(f"Checks failed: {failed}")
    print(f"Warnings: {len(warnings)}")
    print(f"Report saved: {output}")

    if failed == 0:
        print("\nSTRUCTURAL VALIDATION PASSED")
    else:
        print("\nVALIDATION ISSUES REQUIRE FIXING")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

