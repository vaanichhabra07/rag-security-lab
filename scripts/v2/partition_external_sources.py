
import json
import random

from collections import Counter, defaultdict

from scenario_utils import (
    PROCESSED,
    METADATA,
    load_jsonl,
    save_jsonl
)


SEED = 42


def partition_tensor_trust():

    records = load_jsonl(
        PROCESSED
        / "tensor_trust_source_records.jsonl"
    )

    assert len(records) == 1346

    groups = defaultdict(list)

    excluded = []

    for record in records:

        if record["empty_attack"]:

            record["source_partition"] = "excluded"

            excluded.append(record)

            continue

        groups[
            record["normalized_text_hash"]
        ].append(record)

    hashes = sorted(groups)

    rng = random.Random(SEED)

    rng.shuffle(hashes)

    cutoff = int(len(hashes) * 0.70)

    development_hashes = set(
        hashes[:cutoff]
    )

    held_out_hashes = set(
        hashes[cutoff:]
    )

    assert not (
        development_hashes
        & held_out_hashes
    )

    development = []
    held_out = []

    for fingerprint, group in groups.items():

        partition = (

            "development"

            if fingerprint in development_hashes

            else "held_out"
        )

        for record in group:

            record["source_partition"] = (
                partition
            )

            if partition == "development":

                development.append(record)

            else:

                held_out.append(record)

    assert (
        len(development)
        + len(held_out)
        + len(excluded)
    ) == len(records)

    save_jsonl(

        PROCESSED
        / "tensor_trust_development.jsonl",

        development
    )

    save_jsonl(

        PROCESSED
        / "tensor_trust_held_out.jsonl",

        held_out
    )

    save_jsonl(

        PROCESSED
        / "tensor_trust_excluded.jsonl",

        excluded
    )

    return {

        "total_records": len(records),

        "development_records": len(development),

        "held_out_records": len(held_out),

        "excluded_records": len(excluded),

        "development_unique_hashes":
            len(development_hashes),

        "held_out_unique_hashes":
            len(held_out_hashes),

        "normalized_hash_overlap": len(
            development_hashes
            & held_out_hashes
        ),

        "development_objectives": dict(
            Counter(
                r["proposed_objective"]
                for r in development
            )
        ),

        "held_out_objectives": dict(
            Counter(
                r["proposed_objective"]
                for r in held_out
            )
        )
    }


def partition_bipia():

    records = load_jsonl(
        PROCESSED / "bipia_inventory.jsonl"
    )

    assert len(records) == 150

    assert all(
        r["source_split"] in {"train", "test"}
        for r in records
    )

    development = [

        r for r in records

        if r["source_split"] == "train"
    ]

    held_out = [

        r for r in records

        if r["source_split"] == "test"
    ]

    for record in development:

        record["source_partition"] = (
            "development"
        )

    for record in held_out:

        record["source_partition"] = (
            "held_out"
        )

    development_hashes = {

        r["original_text_hash"]

        for r in development
    }

    held_out_hashes = {

        r["original_text_hash"]

        for r in held_out
    }

    assert not (
        development_hashes
        & held_out_hashes
    )

    save_jsonl(

        PROCESSED / "bipia_development.jsonl",

        development
    )

    save_jsonl(

        PROCESSED / "bipia_held_out.jsonl",

        held_out
    )

    return {

        "total_records": len(records),

        "development_records": len(development),

        "held_out_records": len(held_out),

        "exact_hash_overlap": len(
            development_hashes
            & held_out_hashes
        ),

        "original_source_splits_preserved": True
    }


def main():

    tensor_report = partition_tensor_trust()

    bipia_report = partition_bipia()

    report = {

        "dataset_version": "2.0-dev",

        "random_seed": SEED,

        "tensor_trust": tensor_report,

        "bipia": bipia_report,

        "partition_validation_passed": True,

        "semantic_duplicate_review_complete": False,

        "strategy_separation_verified": False,

        "license_verification_complete": False,

        "records_integrated": 0,

        "research_validation_complete": False
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        METADATA
        / "external_partition_statistics.json"
    )

    output.write_text(

        json.dumps(report, indent=2),

        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    print("\nEXTERNAL SOURCE PARTITIONING COMPLETED")


if __name__ == "__main__":
    main()
