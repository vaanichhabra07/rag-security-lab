
import json

from scenario_utils import ROOT, METADATA


def main():

    path = (
        ROOT / "configs"
        / "external_sources_v2.json"
    )

    config = json.loads(
        path.read_text(encoding="utf-8")
    )

    sources = config["sources"]

    distribution = config[
        "proposed_main_dataset_distribution"
    ]

    assert len(sources) == 3

    assert sum(distribution.values()) == 3000

    assert distribution["custom_benign"] == 700

    assert (
        sources["deepset"]["train_records"]
        + sources["deepset"]["test_records"]
        == sources["deepset"]["total_records"]
    )

    assert (
        sources["bipia"]["proposed_samples"]
        == distribution["bipia_derived"]
    )

    assert (
        sources["tensor_trust"]["proposed_samples"]
        == distribution["tensor_trust_derived"]
    )

    report = {

        "sources_reviewed": len(sources),

        "proposed_total_samples": sum(
            distribution.values()
        ),

        "proposed_custom_samples": (
            distribution["custom_benign"]
            + distribution["custom_adversarial"]
        ),

        "proposed_external_samples": (
            distribution["bipia_derived"]
            + distribution["tensor_trust_derived"]
        ),

        "source_status": {

            name: source["status"]

            for name, source in sources.items()
        },

        "registry_validation_passed": True,

        "allocation_revision_applied": False,

        "dataset_license_review_complete": False,

        "actual_samples_reviewed": 0,

        "external_samples_integrated": 0,

        "research_validation_complete": False
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        METADATA
        / "external_source_verification.json"
    )

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    print("\nSOURCE REGISTRY VALIDATION COMPLETED")


if __name__ == "__main__":

    main()
