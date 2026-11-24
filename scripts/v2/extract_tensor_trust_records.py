
import json
import re
import unicodedata

from collections import Counter

from scenario_utils import (
    ROOT,
    PROCESSED,
    METADATA,
    load_jsonl,
    save_jsonl,
    text_hash
)


SOURCE = (
    ROOT / "data" / "v2"
    / "raw" / "tensor-trust"
)


def normalize_text(text):

    text = unicodedata.normalize(
        "NFKC", text
    ).casefold()

    return re.sub(
        r"\s+", " ", text
    ).strip()


def main():

    inventory = json.loads(

        (
            METADATA
            / "tensor_trust_inventory.json"
        ).read_text(encoding="utf-8")
    )

    revision = inventory["source_revision"]

    datasets = {

        "extraction": {

            "filename":
                "extraction_robustness_dataset.jsonl",

            "objective":
                "synthetic_information_disclosure",

            "expected_count": 570
        },

        "hijacking": {

            "filename":
                "hijacking_robustness_dataset.jsonl",

            "objective":
                "instruction_hijacking",

            "expected_count": 776
        }
    }

    records = []

    for dataset_type, config in datasets.items():

        paths = list(
            SOURCE.rglob(config["filename"])
        )

        assert len(paths) == 1

        path = paths[0]

        samples = load_jsonl(path)

        assert len(samples) == config[
            "expected_count"
        ]

        for index, sample in enumerate(samples):

            required_fields = {

                "sample_id",
                "attack",
                "pre_prompt",
                "post_prompt",
                "access_code"
            }

            assert required_fields.issubset(
                sample
            )

            attack = sample["attack"]

            assert isinstance(attack, str)

            normalized = normalize_text(attack)

            source_id = (

                f"TT_{dataset_type.upper()}_"
                f"{sample['sample_id']}"
            )

            records.append({

                "source": "tensor_trust",

                "source_id": source_id,

                "original_sample_id":
                    sample["sample_id"],

                "source_dataset": dataset_type,

                "source_revision": revision,

                "source_file": path.relative_to(
                    SOURCE
                ).as_posix(),

                "source_record_index": index,

                "source_url": (
                    "https://github.com/"
                    "HumanCompatibleAI/"
                    "tensor-trust-data"
                ),

                "source_license": None,

                "license_verified": False,

                "original_text_hash":
                    text_hash(attack),

                "normalized_text_hash":
                    text_hash(normalized),

                "attack_text_length": len(attack),

                "empty_attack":
                    not bool(normalized),

                "original_objective":
                    dataset_type,

                "proposed_objective":
                    config["objective"],

                "verified_attack_family": None,

                "verified_attack_subcategory": None,

                "source_partition": "unassigned",

                "adaptation_status": "not_started",

                "annotation_status": "pending",

                "evaluation_ready": False
            })

    assert len(records) == 1346

    assert len({
        r["source_id"]
        for r in records
    }) == len(records)

    hash_counts = Counter(

        r["normalized_text_hash"]

        for r in records
    )

    for record in records:

        record["duplicate_group_size"] = (
            hash_counts[
                record["normalized_text_hash"]
            ]
        )

    extraction_hashes = {

        r["normalized_text_hash"]

        for r in records

        if r["source_dataset"] == "extraction"
    }

    hijacking_hashes = {

        r["normalized_text_hash"]

        for r in records

        if r["source_dataset"] == "hijacking"
    }

    save_jsonl(

        PROCESSED
        / "tensor_trust_source_records.jsonl",

        records
    )

    report = {

        "total_source_records": len(records),

        "source_revision": revision,

        "dataset_distribution": dict(
            Counter(
                r["source_dataset"]
                for r in records
            )
        ),

        "unique_source_ids": len({
            r["source_id"]
            for r in records
        }),

        "unique_original_text_hashes": len({
            r["original_text_hash"]
            for r in records
        }),

        "unique_normalized_text_hashes":
            len(hash_counts),

        "duplicate_records_beyond_first": (
            len(records) - len(hash_counts)
        ),

        "cross_dataset_text_overlap": len(
            extraction_hashes
            & hijacking_hashes
        ),

        "empty_attack_records": sum(
            r["empty_attack"]
            for r in records
        ),

        "planned_tensor_trust_samples": 350,

        "source_records_extracted": True,

        "attack_family_mapping_verified": False,

        "dataset_license_verified": False,

        "manual_review_complete": False,

        "records_integrated": 0,

        "research_validation_complete": False
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        METADATA
        / "tensor_trust_extraction_report.json"
    )

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    print("\nTENSOR TRUST EXTRACTION COMPLETED")


if __name__ == "__main__":
    main()
