
import json

from collections import Counter, defaultdict

from scenario_utils import ROOT, METADATA


SOURCE = (
    ROOT / "data" / "v2"
    / "raw" / "tensor-trust"
)


def inspect_file(path):

    total_records = 0
    invalid_records = 0

    field_counts = Counter()
    field_types = defaultdict(Counter)
    empty_values = Counter()
    field_lengths = defaultdict(list)

    with path.open(
        "r",
        encoding="utf-8-sig"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            try:

                record = json.loads(line)

            except json.JSONDecodeError:

                invalid_records += 1
                continue

            if not isinstance(record, dict):

                invalid_records += 1
                continue

            total_records += 1

            for key, value in record.items():

                field_counts[key] += 1

                field_types[key][
                    type(value).__name__
                ] += 1

                if value is None or value == "":

                    empty_values[key] += 1

                if isinstance(value, str):

                    field_lengths[key].append(
                        len(value)
                    )

    fields = {}

    for key in sorted(field_counts):

        lengths = field_lengths[key]

        fields[key] = {

            "present_records": field_counts[key],

            "missing_records": (
                total_records - field_counts[key]
            ),

            "empty_values": empty_values[key],

            "data_types": dict(
                field_types[key]
            ),

            "minimum_text_length": (
                min(lengths) if lengths else None
            ),

            "maximum_text_length": (
                max(lengths) if lengths else None
            )
        }

    return {

        "filename": path.name,

        "relative_path": path.relative_to(
            SOURCE
        ).as_posix(),

        "total_records": total_records,

        "invalid_records": invalid_records,

        "fields": fields
    }


def main():

    files = sorted(
        SOURCE.rglob("*.jsonl")
    )

    assert len(files) == 3, (
        "Expected three Tensor Trust JSONL files"
    )

    reports = [

        inspect_file(path)

        for path in files
    ]

    report = {

        "source": "Tensor Trust",

        "files_inspected": len(reports),

        "total_records": sum(
            r["total_records"]
            for r in reports
        ),

        "invalid_records": sum(
            r["invalid_records"]
            for r in reports
        ),

        "datasets": reports,

        "schema_inspection_complete": True,

        "semantic_field_mapping_verified": False,

        "dataset_license_verified": False,

        "records_integrated": 0
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        METADATA
        / "tensor_trust_schema_report.json"
    )

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    for dataset in reports:

        print("\n" + "=" * 60)

        print(
            "Dataset:",
            dataset["filename"]
        )

        print(
            "Records:",
            dataset["total_records"]
        )

        print(
            "Invalid records:",
            dataset["invalid_records"]
        )

        print("\nAvailable fields:")

        for field, details in dataset[
            "fields"
        ].items():

            print(
                f"  {field}: "
                f"{details['data_types']}"
            )

    print("\n" + "=" * 60)

    print(
        "Total records:",
        report["total_records"]
    )

    print(
        "Invalid records:",
        report["invalid_records"]
    )

    print("\nTENSOR TRUST SCHEMA INSPECTION COMPLETED")


if __name__ == "__main__":
    main()
