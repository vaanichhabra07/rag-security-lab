
import csv
import json

from collections import Counter

from scenario_utils import (
    PROCESSED,
    METADATA,
    load_jsonl
)


def main():

    inventory = load_jsonl(
        PROCESSED / "bipia_inventory.jsonl"
    )

    assert len(inventory) == 150

    rows = []

    for record in inventory:

        rows.append({

            "source_id": record["source_id"],

            "source_split": record["source_split"],

            "source_category": record["source_category"],

            "original_text_hash": record[
                "original_text_hash"
            ],

            "proposed_family": "indirect_injection",

            "verified_family": "",

            "verified_objective": "",

            "source_license": "",

            "license_verified": False,

            "adaptation_required": "",

            "ground_truth_verified": False,

            "review_status": "pending",

            "review_notes": ""
        })

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        METADATA / "bipia_v2_review.csv"
    )

    if output.exists():

        raise FileExistsError(
            "Review file already exists. "
            "Preserve existing annotations."
        )

    with output.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0])
        )

        writer.writeheader()

        writer.writerows(rows)

    report = {

        "templates_pending_review": len(rows),

        "source_split_distribution": dict(
            Counter(
                r["source_split"]
                for r in rows
            )
        ),

        "approved_templates": 0,

        "manual_review_complete": False,

        "license_verification_complete": False,

        "evaluation_ready": False
    }

    (
        METADATA / "bipia_v2_review_statistics.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    print("\nBIPIA REVIEW SHEET CREATED")


if __name__ == "__main__":
    main()
