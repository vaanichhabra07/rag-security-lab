
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

SEED = 42


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

    rng = random.Random(SEED)

    templates = load_jsonl("bipia_classified.jsonl")
    plan = load_jsonl("adversarial_sample_plan.jsonl")

    template_groups = defaultdict(list)
    slot_groups = defaultdict(list)

    families = [
        "obfuscation",
        "indirect_injection"
    ]

    quotas = {
        "train": 60,
        "validation": 30,
        "test_seen": 30,
        "test_unseen": 30
    }

    for template in templates:
        key = (
            template["candidate_attack_family"],
            template["source_split"]
        )

        template_groups[key].append(template)

    for slot in plan:
        key = (
            slot["attack_family"],
            slot["split"]
        )

        slot_groups[key].append(slot)

    selected = []

    for family in families:

        for split, quota in quotas.items():

            partition = (
                "test"
                if split == "test_unseen"
                else "train"
            )

            candidates = template_groups[
                (family, partition)
            ]

            if not candidates:
                raise ValueError(
                    f"No candidates: {family}, {partition}"
                )

            slots = rng.sample(
                slot_groups[(family, split)],
                quota
            )

            for index, slot in enumerate(slots):

                template = candidates[
                    index % len(candidates)
                ]

                selected.append({
                    **slot,
                    "source": "BIPIA",
                    "source_id": template["template_id"],
                    "source_category":
                        template["source_category"],
                    "source_split":
                        template["source_split"],
                    "template_hash":
                        template["template_hash"],
                    "review_status": "pending",
                    "status": "selected_for_review"
                })

    assert len(selected) == 300

    assert len({
        r["planned_sample_id"]
        for r in selected
    }) == 300

    output = (
        PROCESSED / "bipia_selected_samples.jsonl"
    )

    with output.open("w", encoding="utf-8") as f:

        for record in selected:
            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    print("BIPIA SAMPLE SELECTION COMPLETE")
    print(f"Selected samples: {len(selected)}")

    print("\nFamily distribution:")

    for family, count in Counter(
        r["attack_family"] for r in selected
    ).items():
        print(f"{family}: {count}")

    print("\nPartition distribution:")

    for split, count in Counter(
        r["split"] for r in selected
    ).items():
        print(f"{split}: {count}")

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
