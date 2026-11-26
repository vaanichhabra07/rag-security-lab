
import sys
import json

from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.defenses.rule_based import (
    RuleBasedDefense
)


PROCESSED = ROOT / "data/v2/processed"

METADATA = ROOT / "data/v2/metadata"


def load_jsonl(path):

    with path.open(
        encoding="utf-8-sig"
    ) as file:

        return [

            json.loads(line)

            for line in file

            if line.strip()
        ]


def divide(a, b):

    return a / b if b else 0.0


def main():

    plan = load_jsonl(

        PROCESSED
        / "adversarial_sample_plan_revised.jsonl"
    )

    train_documents = {

        record["document_id"]

        for record in plan

        if record["split"] == "train"
    }

    benign = load_jsonl(

        PROCESSED / "benign_samples.jsonl"
    )

    direct = load_jsonl(

        PROCESSED / "custom_direct_samples.jsonl"
    )

    benign = [

        record

        for record in benign

        if record["document_id"] in train_documents

        and not record.get("conversation_history")

        and record.get("user_input")
    ]

    adversarial = [

        record

        for record in direct

        if record["split"] == "train"

        and record["document_id"] in train_documents

        and record["materialization_status"]
        == "complete"
    ]

    assert benign, "No eligible benign records"

    assert adversarial, "No eligible attack records"

    samples = benign + adversarial

    defense = RuleBasedDefense()

    results = []

    tp = fp = tn = fn = 0

    for sample in samples:

        scan = defense.scan(

            text=sample["user_input"],

            channel="user_input"
        )

        actual = sample["label"] == "adversarial"

        predicted = scan["detected"]

        if actual and predicted:

            tp += 1

        elif actual and not predicted:

            fn += 1

        elif not actual and predicted:

            fp += 1

        else:

            tn += 1

        results.append({

            "sample_id": sample["sample_id"],

            "actual_label": sample["label"],

            "attack_family": sample["attack_family"],

            "detected": predicted,

            "matched_rules": scan["matched_rules"],

            "detection_latency_ms":
                scan["detection_latency_ms"]
        })

    precision = divide(tp, tp + fp)

    recall = divide(tp, tp + fn)

    fpr = divide(fp, fp + tn)

    accuracy = divide(tp + tn, len(samples))

    f1 = divide(

        2 * precision * recall,

        precision + recall
    )

    report = {

        "defense": "D1",

        "evaluation_scope":
            "training_development_only",

        "evaluation_component":
            "user_input_detection",

        "formal_test_evaluation": False,

        "total_samples": len(samples),

        "benign_samples": len(benign),

        "adversarial_samples": len(adversarial),

        "true_positives": tp,

        "false_positives": fp,

        "true_negatives": tn,

        "false_negatives": fn,

        "precision": round(precision, 4),

        "detection_recall": round(recall, 4),

        "false_positive_rate": round(fpr, 4),

        "accuracy": round(accuracy, 4),

        "f1_score": round(f1, 4),

        "average_detection_latency_ms": round(

            mean(

                result["detection_latency_ms"]

                for result in results
            ),

            4
        ),

        "attack_success_rate": None,

        "research_validation_complete": False
    }

    METADATA.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        METADATA
        / "d1_development_evaluation.json"
    ).write_text(

        json.dumps(report, indent=2),

        encoding="utf-8"
    )

    with (

        PROCESSED
        / "d1_development_results.jsonl"

    ).open("w", encoding="utf-8") as file:

        for result in results:

            file.write(

                json.dumps(result) + "\n"
            )

    print(json.dumps(report, indent=2))

    print("\nD1 DEVELOPMENT EVALUATION COMPLETED")


if __name__ == "__main__":

    main()
