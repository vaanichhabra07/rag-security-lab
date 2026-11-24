
import json

from collections import Counter

from scenario_utils import (
    ROOT,
    PROCESSED,
    METADATA,
    load_jsonl,
    save_jsonl,
    text_hash
)


def main():

    source_path = (
        ROOT / "data"
        / "processed"
        / "bipia_templates.jsonl"
    )

    templates = load_jsonl(source_path)

    assert len(templates) == 150

    inventory = []

    for template in templates:

        text = template["attack_text"]

        assert isinstance(text, str)

        assert text.strip()

        inventory.append({

            "source": "BIPIA",

            "source_id": template["template_id"],

            "source_split": template["source_split"],

            "source_category": template["source_category"],

            "source_file": template["source_file"],

            "original_text_hash": text_hash(text),

            "original_text_length": len(text),

            "original_attack_channel": "retrieved_context",

            "proposed_attack_family": "indirect_injection",

            "source_url": (
                "https://github.com/microsoft/BIPIA"
            ),

            "source_version": None,

            "source_license": None,

            "license_verified": False,

            "record_review_status": "pending",

            "adaptation_status": "not_started",

            "evaluation_ready": False
        })

    assert len({
        r["source_id"] for r in inventory
    }) == 150

    development_hashes = {

        r["original_text_hash"]

        for r in inventory

        if r["source_split"] == "train"
    }

    test_hashes = {

        r["original_text_hash"]

        for r in inventory

        if r["source_split"] == "test"
    }

    plan = load_jsonl(
        PROCESSED
        / "adversarial_sample_plan_revised.jsonl"
    )

    bipia_slots = [

        r for r in plan

        if r["planned_source_group"] == "bipia"
    ]

    assert len(bipia_slots) == 500

    assert all(
        r["attack_family"] == "indirect_injection"
        for r in bipia_slots
    )

    save_jsonl(
        PROCESSED / "bipia_inventory.jsonl",
        inventory
    )

    report = {

        "available_source_templates": len(inventory),

        "unique_text_hashes": len({
            r["original_text_hash"]
            for r in inventory
        }),

        "source_split_distribution": dict(
            Counter(
                r["source_split"]
                for r in inventory
            )
        ),

        "source_category_distribution": dict(
            Counter(
                r["source_category"]
                for r in inventory
            )
        ),

        "exact_source_split_overlap": len(
            development_hashes & test_hashes
        ),

        "planned_bipia_samples": len(bipia_slots),

        "planned_split_distribution": dict(
            Counter(
                r["split"]
                for r in bipia_slots
            )
        ),

        "record_review_complete": False,

        "dataset_license_verified": False,

        "samples_integrated": 0,

        "inventory_validation_passed": True,

        "research_validation_complete": False
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        METADATA / "bipia_inventory_report.json"
    )

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    print("\nBIPIA INVENTORY COMPLETED")


if __name__ == "__main__":
    main()
