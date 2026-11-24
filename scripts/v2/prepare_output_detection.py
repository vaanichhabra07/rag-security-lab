
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

    source = (

        ROOT / "data" / "v2" / "raw"
        / "tensor-trust"
        / "detecting-extractions" / "v1"
        / "prompt_extraction_detection.jsonl"
    )

    samples = load_jsonl(source)

    assert len(samples) == 230

    records = []

    for sample in samples:

        label = sample["is_prompt_extraction"]

        assert isinstance(label, bool)

        assert isinstance(
            sample["llm_output"], str
        )

        records.append({

            "sample_id": (
                f"TT_OUTPUT_{sample['sample_id']}"
            ),

            "original_sample_id":
                sample["sample_id"],

            "source": "tensor_trust",

            "source_dataset":
                "prompt_extraction_detection",

            "output_hash": text_hash(
                sample["llm_output"]
            ),

            "is_prompt_extraction": label,

            "evaluation_component":
                "D4_output_validator",

            "benchmark_role":
                "supplementary_output_detection",

            "included_in_main_dataset": False,

            "license_verified": False,

            "evaluation_ready": False
        })

    assert len({
        r["sample_id"]
        for r in records
    }) == 230

    save_jsonl(

        PROCESSED
        / "supplementary_output_detection.jsonl",

        records
    )

    report = {

        "total_records": len(records),

        "label_distribution": dict(
            Counter(
                str(r["is_prompt_extraction"])
                for r in records
            )
        ),

        "included_in_main_dataset": False,

        "main_dataset_target": 3000,

        "evaluation_component":
            "D4_output_validator",

        "license_verified": False,

        "evaluation_ready": False
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        METADATA
        / "output_detection_statistics.json"
    ).write_text(

        json.dumps(report, indent=2),

        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    print("\nOUTPUT DETECTION PREPARATION COMPLETED")


if __name__ == "__main__":
    main()
