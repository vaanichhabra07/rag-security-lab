
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "configs" / "dataset_v2.json"

OUTPUT_PATH = (
    ROOT / "data" / "v2" / "metadata"
    / "configuration_validation.json"
)


def main():

    with CONFIG_PATH.open(
        encoding="utf-8"
    ) as f:
        config = json.load(f)

    checks = []

    def check(name, condition):

        passed = bool(condition)

        checks.append({
            "check": name,
            "passed": passed
        })

        status = "PASS" if passed else "FAIL"

        print(f"[{status}] {name}")

    print("=" * 60)
    print("DATASET V2 CONFIGURATION VALIDATION")
    print("=" * 60)

    targets = config["sample_targets"]

    total = config["total_samples"]

    benign = targets["benign"]

    adversarial = total - benign

    check(
        "Total sample count is correct",
        sum(targets.values()) == total
    )

    check(
        "Five attack families defined",
        len(targets) - 1 == 5
    )

    check(
        "Corpus document count is correct",
        sum(config["corpus"]["domains"].values())
        == config["corpus"]["total_documents"]
    )

    check(
        "Benign category counts are correct",
        sum(config["benign_categories"].values())
        == benign
    )

    check(
        "Source targets sum to total samples",
        sum(config["source_targets"].values())
        == total
    )

    check(
        "Benign source target is correct",
        config["source_targets"]["custom_benign"]
        == benign
    )

    taxonomy = config["attack_taxonomy"]

    check(
        "Taxonomy matches attack families",
        set(taxonomy.keys())
        == set(targets.keys()) - {"benign"}
    )

    subcategories = [
        category
        for categories in taxonomy.values()
        for category in categories
    ]

    check(
        "No duplicate subcategory names",
        len(subcategories)
        == len(set(subcategories))
    )

    splits = config["splits"]

    split_benign = sum(
        s["benign"] for s in splits.values()
    )

    split_adversarial = sum(
        s["adversarial"] for s in splits.values()
    )

    check(
        "Benign split counts are correct",
        split_benign == benign
    )

    check(
        "Adversarial split counts are correct",
        split_adversarial == adversarial
    )

    check(
        "Seen and unseen test counts match",
        splits["test_seen"]["adversarial"]
        == splits["test_unseen"]["adversarial"]
    )

    check(
        "All six defense configurations defined",
        set(config["defenses"])
        == {f"D{i}" for i in range(6)}
    )

    check(
        "At least two backbone models planned",
        config["experimental_protocol"]
        ["minimum_backbone_models"] >= 2
    )

    check(
        "Pairwise ablation enabled",
        config["experimental_protocol"]
        ["pairwise_ablation"]
    )

    check(
        "Grouped splitting enabled",
        config["experimental_protocol"]
        ["grouped_split"]
    )

    check(
        "External actions disabled",
        not config["safety_constraints"]
        ["external_actions_allowed"]
    )

    check(
        "Only synthetic secrets allowed",
        config["safety_constraints"]
        ["synthetic_secrets_only"]
    )

    test_count = sum(
        splits[name]["benign"]
        + splits[name]["adversarial"]
        for name in [
            "test_seen",
            "test_unseen",
            "test_benign"
        ]
    )

    failed = sum(
        not item["passed"] for item in checks
    )

    report = {
        "dataset_version": config["dataset_version"],
        "planned_total_samples": total,
        "planned_benign_samples": benign,
        "planned_adversarial_samples": adversarial,
        "planned_documents": config["corpus"]
        ["total_documents"],
        "attack_families": len(taxonomy),
        "attack_subcategories": len(subcategories),
        "attack_objectives": len(
            config["attack_objectives"]
        ),
        "planned_test_samples": test_count,
        "planned_main_evaluations": (
            test_count
            * len(config["defenses"])
            * config["experimental_protocol"]
            ["minimum_backbone_models"]
        ),
        "checks_passed": len(checks) - failed,
        "checks_failed": failed,
        "configuration_valid": failed == 0,
        "dataset_generated": False,
        "research_validated": False,
        "checks": checks
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)

    for key, value in report.items():
        if key != "checks":
            print(f"{key}: {value}")

    if failed:
        raise SystemExit(
            "\nCONFIGURATION VALIDATION FAILED"
        )

    print("\nCONFIGURATION VALIDATION PASSED")


if __name__ == "__main__":
    main()
