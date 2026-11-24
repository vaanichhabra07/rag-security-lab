
import json
import hashlib

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "v2" / "processed"

METADATA = ROOT / "data" / "v2" / "metadata"


def load_jsonl(path):

    with path.open(encoding="utf-8") as f:

        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save_jsonl(path, records):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open("w", encoding="utf-8") as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


def text_hash(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def create_base_sample(slot, qa):

    return {
        "sample_id": slot["planned_sample_id"],

        "qa_id": qa["qa_id"],

        "document_id": qa["document_id"],

        "domain": qa["domain"],

        "question": qa["question"],

        "user_input": qa["question"],

        "clean_context": qa["clean_context"],

        "modified_context": qa["clean_context"],

        "conversation_history": [],

        "label": "adversarial",

        "benign_category": None,

        "attack_family": slot["attack_family"],

        "attack_subcategory": None,

        "attack_channel": None,

        "attack_template_id": None,

        "attack_objective": None,

        "attack_goal": None,

        "success_criteria": None,

        "expected_answer": qa["expected_answer"],

        "expected_behavior": (
            "answer_legitimate_question"
        ),

        "answerable": True,

        "language": "en",

        "source": slot["planned_source_group"],

        "source_id": None,

        "split": slot["split"],

        "pair_id": slot["pair_id"],

        "template_partition": (
            slot["template_partition"]
        ),

        "novelty": (
            "unseen"
            if slot["split"] == "test_unseen"
            else "seen"
            if slot["split"] == "test_seen"
            else "not_applicable"
        ),

        "document_format": "txt",

        "annotation_status": "pending",

        "materialization_status": "pending",

        "evaluation_ready": False,

        "dataset_version": "2.0-dev"
    }
