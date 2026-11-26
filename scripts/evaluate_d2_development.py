
import sys
import json
import time
import random

from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.defenses.llm_detector import LLMDetector

from src.defenses.rule_based import RuleBasedDefense


PROCESSED = ROOT / "data/v2/processed"

METADATA = ROOT / "data/v2/metadata"

SAMPLES_PER_CLASS = 10

RANDOM_SEED = 42

REQUEST_DELAY_SECONDS = 3


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

    return a / b if b else None


def calculate_metrics(results, detector_name):

    valid = [

        row for row in results

        if row[detector_name]["detected"]
        is not None
    ]

    tp = fp = tn = fn = 0

    for row in valid:

        actual = (
            row["actual_label"] == "adversarial"
        )

        predicted = (
            row[detector_name]["detected"]
        )

        if actual and predicted:

            tp += 1

        elif actual and not predicted:

            fn += 1

        elif not actual and predicted:

            fp += 1

        else:

            tn += 1

    precision = divide(tp, tp + fp)

    recall = divide(tp, tp + fn)

    fpr = divide(fp, fp + tn)

    f1 = divide(
        2 * tp,
        2 * tp + fp + fn
    )

    return {

        "total_samples": len(results),

        "valid_classifications": len(valid),

        "classification_errors":
            len(results) - len(valid),

        "true_positives": tp,

        "false_positives": fp,

        "true_negatives": tn,

        "false_negatives": fn,

        "precision": precision,

        "recall": recall,

        "false_positive_rate": fpr,

        "f1_score": f1,

        "accuracy": divide(
            tp + tn,
            len(valid)
        ),

        "average_detection_latency_ms": (

            mean([

                row[detector_name][
                    "detection_latency_ms"
                ]

                for row in valid
            ])

            if valid else None
        )
    }


def main():

    plan = load_jsonl(

        PROCESSED
        / "adversarial_sample_plan_revised.jsonl"
    )

    train_documents = {

        row["document_id"]

        for row in plan

        if row["split"] == "train"
    }

    benign = load_jsonl(

        PROCESSED / "benign_samples.jsonl"
    )

    adversarial = load_jsonl(

        PROCESSED / "custom_direct_samples.jsonl"
    )

    benign = [

        row for row in benign

        if row["document_id"] in train_documents

        and not row.get("conversation_history")

        and row.get("user_input")
    ]

    adversarial = [

        row for row in adversarial

        if row["split"] == "train"

        and row["document_id"] in train_documents

        and row["materialization_status"]
        == "complete"
    ]

    rng = random.Random(RANDOM_SEED)

    count = min(

        SAMPLES_PER_CLASS,

        len(benign),

        len(adversarial)
    )

    if count == 0:

        raise ValueError(
            "No eligible development samples"
        )

    samples = (

        rng.sample(benign, count)

        + rng.sample(adversarial, count)
    )

    rng.shuffle(samples)

    d1 = RuleBasedDefense()

    d2 = LLMDetector()

    results = []

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (

        PROCESSED
        / "d2_development_results.jsonl"
    )

    with output_file.open(
        "w", encoding="utf-8"
    ) as file:

        for index, sample in enumerate(
            samples, 1
        ):

            text = sample["user_input"]

            result = {

                "sample_id":
                    sample["sample_id"],

                "actual_label":
                    sample["label"],

                "attack_family":
                    sample["attack_family"],

                "D1": d1.scan(text),

                "D2": d2.scan(text)
            }

            results.append(result)

            file.write(

                json.dumps(result) + "\n"
            )

            file.flush()

            print(

                f"{index}/{len(samples)}",

                "| Actual:", sample["label"],

                "| D1:",
                result["D1"]["detected"],

                "| D2:",
                result["D2"]["label"]
            )

            if index != len(samples):

                time.sleep(
                    REQUEST_DELAY_SECONDS
                )

    report = {

        "evaluation_scope":
            "training_development_only",

        "formal_test_evaluation": False,

        "random_seed": RANDOM_SEED,

        "total_samples": len(samples),

        "benign_samples": count,

        "adversarial_samples": count,

        "classifier_model": d2.model,

        "D1": calculate_metrics(
            results, "D1"
        ),

        "D2": calculate_metrics(
            results, "D2"
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
        / "d2_development_evaluation.json"

    ).write_text(

        json.dumps(report, indent=2),

        encoding="utf-8"
    )

    print("\n" + "=" * 60)

    print(json.dumps(report, indent=2))

    print(
        "\nD1 VS D2 DEVELOPMENT EVALUATION COMPLETED"
    )


if __name__ == "__main__":

    main()
