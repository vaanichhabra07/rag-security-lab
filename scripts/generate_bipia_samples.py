
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def load_jsonl(filename):
    with (PROCESSED / filename).open(
        "r", encoding="utf-8-sig"
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():

    benign = load_jsonl("benign_samples.jsonl")
    templates = load_jsonl("bipia_classified.jsonl")
    selected = load_jsonl("bipia_selected_samples.jsonl")

    qa_lookup = {
        r["qa_id"]: r for r in benign
    }

    template_lookup = {
        r["template_id"]: r for r in templates
    }

    records = []

    for slot in selected:

        template = template_lookup[slot["source_id"]]

        assert (
            template["source_split"]
            == slot["template_partition"]
        )

        record = qa_lookup[slot["qa_id"]].copy()

        record.update({
            "sample_id": slot["planned_sample_id"],
            "label": "adversarial",
            "attack_family": slot["attack_family"],
            "attack_channel": "retrieved_context",
            "attack_template_id": template["template_id"],
            "split": slot["split"],
            "novelty": (
                "unseen"
                if slot["split"] == "test_unseen"
                else "seen"
                if slot["split"] == "test_seen"
                else "development"
            ),
            "source": "BIPIA_derived",
            "source_id": template["template_id"],
            "source_category": template["source_category"],
            "source_split": template["source_split"],
            "template_hash": template["template_hash"],
            "attack_goal": None,
            "success_criteria": None,
            "review_status": "pending",
            "evaluation_ready": False
        })

        # Preserve the original benchmark text separately
        # until its suitability and evaluation goal are reviewed.
        record["benchmark_template"] = template["attack_text"]

        records.append(record)

    assert len(records) == 300

    assert len({
        r["sample_id"] for r in records
    }) == 300

    output = PROCESSED / "bipia_review_samples.jsonl"

    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    print("=" * 50)
    print("BIPIA REVIEW RECORDS GENERATED")
    print("=" * 50)

    print(f"Total records: {len(records)}")

    for family, count in Counter(
        r["attack_family"] for r in records
    ).items():
        print(f"{family}: {count}")

    print("\nEvaluation-ready records: 0")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
