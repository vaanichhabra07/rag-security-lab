
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


plan = load_jsonl("adversarial_sample_plan.jsonl")
bipia = load_jsonl("bipia_selected_samples.jsonl")

bipia_ids = {
    r["planned_sample_id"] for r in bipia
}

custom = [
    r for r in plan
    if r["planned_sample_id"] not in bipia_ids
]

custom_ids = {
    r["planned_sample_id"] for r in custom
}

assert len(custom) == 950
assert len(bipia_ids) == 300
assert not custom_ids.intersection(bipia_ids)

output = PROCESSED / "custom_sample_plan.jsonl"

with output.open("w", encoding="utf-8") as f:
    for record in custom:
        f.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )

print("CUSTOM SAMPLE ALLOCATION COMPLETE")
print(f"Custom allocations: {len(custom)}")
print(f"BIPIA allocations: {len(bipia_ids)}")
print("Overlapping allocations: 0")

print("\nCustom family distribution:")

for family, count in Counter(
    r["attack_family"] for r in custom
).items():
    print(f"{family}: {count}")

print(f"\nSaved: {output}")
