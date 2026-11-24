
import json
import hashlib
import subprocess

from collections import Counter

from scenario_utils import ROOT, METADATA


SOURCE = (
    ROOT / "data" / "v2"
    / "raw" / "tensor-trust"
)


def calculate_hash(path):

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):

            digest.update(block)

    return digest.hexdigest()


def main():

    if not SOURCE.exists():

        raise FileNotFoundError(
            "Tensor Trust dataset directory missing"
        )

    revision = subprocess.check_output(

        [
            "git",
            "-C",
            str(SOURCE),
            "rev-parse",
            "HEAD"
        ],

        text=True

    ).strip()

    data_extensions = {

        ".json",
        ".jsonl",
        ".csv",
        ".tsv",
        ".gz",
        ".parquet",
        ".zip"
    }

    data_files = []

    documentation_files = []

    for path in sorted(SOURCE.rglob("*")):

        if not path.is_file():
            continue

        relative = path.relative_to(SOURCE)

        if ".git" in relative.parts:
            continue

        if path.suffix.lower() in data_extensions:

            data_files.append({

                "relative_path": str(relative),

                "extension": path.suffix.lower(),

                "size_bytes": path.stat().st_size,

                "sha256": calculate_hash(path)
            })

        if any(
            name in path.name.lower()

            for name in [
                "license",
                "readme",
                "copying"
            ]
        ):

            documentation_files.append(
                str(relative)
            )

    report = {

        "source": "Tensor Trust",

        "repository": (
            "https://github.com/"
            "HumanCompatibleAI/tensor-trust-data"
        ),

        "source_revision": revision,

        "data_files_found": len(data_files),

        "file_format_distribution": dict(
            Counter(
                r["extension"]
                for r in data_files
            )
        ),

        "total_data_size_bytes": sum(
            r["size_bytes"]
            for r in data_files
        ),

        "data_files": data_files,

        "documentation_files": documentation_files,

        "planned_main_benchmark_samples": 350,

        "planned_family_distribution": {

            "direct_injection": 200,

            "instruction_manipulation": 150
        },

        "dataset_schema_verified": False,

        "dataset_license_verified": False,

        "records_integrated": 0,

        "research_validation_complete": False
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        METADATA / "tensor_trust_inventory.json"
    )

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    print("\nTENSOR TRUST INVENTORY COMPLETED")


if __name__ == "__main__":
    main()
