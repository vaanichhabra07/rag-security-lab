
import json

from collections import Counter

from scenario_utils import (
    PROCESSED,
    METADATA,
    load_jsonl
)


def main():

    plan = load_jsonl(

        PROCESSED
        / "adversarial_sample_plan_revised.jsonl"
    )

    external_slots = [

        r for r in plan

        if r["planned_source_group"] in {
            "bipia",
            "tensor_trust"
        }
    ]

    assert len(external_slots) == 850

    source_counts = Counter(

        r["planned_source_group"]

        for r in external_slots
    )

    assert dict(source_counts) == {

        "bipia": 500,

        "tensor_trust": 350
    }

    results = {}

    for source in ["bipia", "tensor_trust"]:

        slots = [

            r for r in external_slots

            if r["planned_source_group"] == source
        ]

        development = load_jsonl(

            PROCESSED
            / f"{source}_development.jsonl"
        )

        held_out = load_jsonl(

            PROCESSED
            / f"{source}_held_out.jsonl"
        )

        required_development = sum(

            r["template_partition"] == "development"

            for r in slots
        )

        required_held_out = sum(

            r["template_partition"] == "held_out"

            for r in slots
        )

        hash_field = (

            "normalized_text_hash"

            if source == "tensor_trust"

            else "original_text_hash"
        )

        unique_development = len({

            r[hash_field]

            for r in development
        })

        unique_held_out = len({

            r[hash_field]

            for r in held_out
        })

        results[source] = {

            "planned_samples": len(slots),

            "required_development_slots":
                required_development,

            "required_held_out_slots":
                required_held_out,

            "available_development_records":
                len(development),

            "available_held_out_records":
                len(held_out),

            "unique_development_texts":
                unique_development,

            "unique_held_out_texts":
                unique_held_out,

            "template_reuse_required": (

                unique_development
                < required_development

                or unique_held_out
                < required_held_out
            ),

            "source_pools_nonempty": bool(
                development and held_out
            ),

            "family_mapping_verified": False,

            "license_verified": False,

            "ready_for_final_integration": False
        }

    report = {

        "planned_external_samples": 850,

        "source_readiness": results,

        "allocation_counts_verified": True,

        "actual_samples_integrated": 0,

        "final_integration_approved": False
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        METADATA
        / "external_integration_readiness.json"
    ).write_text(

        json.dumps(report, indent=2),

        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    print("\nEXTERNAL READINESS CHECK COMPLETED")


if __name__ == "__main__":
    main()
