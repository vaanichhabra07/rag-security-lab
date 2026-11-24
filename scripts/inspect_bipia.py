
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIPIA_DIR = ROOT / "data" / "raw" / "bipia" / "benchmark"
OUTPUT_DIR = ROOT / "data" / "metadata"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

report = {
    "datasets": {},
    "attack_files": {}
}


def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def inspect_datasets():
    print("\n" + "=" * 60)
    print("BIPIA DATASET INSPECTION")
    print("=" * 60)

    total_samples = 0

    for task in ["email", "code", "table", "qa", "abstract"]:
        print(f"\nTASK: {task.upper()}")
        report["datasets"][task] = {}

        for split in ["train", "test"]:
            path = BIPIA_DIR / task / f"{split}.jsonl"

            if not path.exists():
                print(f"  {split}: File not available")
                report["datasets"][task][split] = {
                    "available": False
                }
                continue

            records = load_jsonl(path)
            total_samples += len(records)

            fields = sorted({
                key
                for record in records
                if isinstance(record, dict)
                for key in record
            })

            print(f"  {split}: {len(records)} samples")
            print(f"  Fields: {fields}")

            report["datasets"][task][split] = {
                "available": True,
                "sample_count": len(records),
                "fields": fields
            }

            if records:
                print("  First sample preview:")

                sample = json.dumps(
                    records[0],
                    indent=2,
                    ensure_ascii=True
                )

                print(sample[:600])

    report["total_context_records"] = total_samples

    print(f"\nTotal context records: {total_samples}")


def inspect_attacks():
    print("\n" + "=" * 60)
    print("ATTACK TEMPLATE INSPECTION")
    print("=" * 60)

    for path in sorted(BIPIA_DIR.glob("*attack*.json")):

        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)

        print(f"\nFILE: {path.name}")
        print(f"Data type: {type(data).__name__}")

        info = {
            "data_type": type(data).__name__
        }

        if isinstance(data, dict):
            categories = list(data.keys())

            print(f"Top-level entries: {len(categories)}")
            print(f"Keys: {categories}")

            info["top_level_keys"] = categories
            info["top_level_count"] = len(categories)
            info["category_sizes"] = {}

            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    count = len(value)
                    info["category_sizes"][key] = count
                    print(f"  {key}: {count} entries")
                else:
                    print(f"  {key}: {type(value).__name__}")

        elif isinstance(data, list):
            info["entry_count"] = len(data)
            print(f"Total entries: {len(data)}")

            if data and isinstance(data[0], dict):
                info["sample_fields"] = list(data[0].keys())
                print(f"Fields: {info['sample_fields']}")

        report["attack_files"][path.name] = info


def save_report():
    output_path = OUTPUT_DIR / "bipia_inspection.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=True)

    print("\n" + "=" * 60)
    print(f"Report saved: {output_path}")
    print("INSPECTION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":

    if not BIPIA_DIR.exists():
        raise FileNotFoundError(
            f"BIPIA benchmark not found: {BIPIA_DIR}"
        )

    inspect_datasets()
    inspect_attacks()
    save_report()
