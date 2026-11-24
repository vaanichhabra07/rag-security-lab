
import json
import hashlib
import re

from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parents[2]

OLD_CORPUS = ROOT / "data" / "corpus"

NEW_CORPUS = ROOT / "data" / "v2" / "corpus"

POLICY_FILE = (
    ROOT / "configs"
    / "institutional_policies_v2.json"
)

PROCESSED = (
    ROOT / "data" / "v2" / "processed"
)

METADATA = (
    ROOT / "data" / "v2" / "metadata"
)

OLD_BENIGN = (
    ROOT / "data" / "processed"
    / "benign_samples.jsonl"
)


def load_jsonl(path):

    with path.open(encoding="utf-8") as f:

        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save_jsonl(path, records):

    with path.open("w", encoding="utf-8") as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


def slug(text):

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        text.lower()
    ).strip("_")


def sha256(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def main():

    for directory in [
        NEW_CORPUS,
        PROCESSED,
        METADATA
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    original_documents = sorted(
        OLD_CORPUS.rglob("*.txt")
    )

    assert len(original_documents) == 50

    original_hashes = {
        str(p.relative_to(OLD_CORPUS)):
        sha256(p)
        for p in original_documents
    }

    # Preserve the original corpus.
    for source in original_documents:

        relative = source.relative_to(
            OLD_CORPUS
        )

        destination = NEW_CORPUS / relative

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if destination.exists():

            assert sha256(destination) == sha256(
                source
            ), f"Existing copy differs: {destination}"

        else:

            destination.write_bytes(
                source.read_bytes()
            )

    with POLICY_FILE.open(
        encoding="utf-8"
    ) as f:

        policies = json.load(f)

    old_benign = load_jsonl(OLD_BENIGN)

    assert len(old_benign) == 250

    old_document_ids = {
        record["document_id"]
        for record in old_benign
    }

    new_qa = []

    manifest = []

    document_counter = 0

    for domain, entries in policies.items():

        assert len(entries) == 5

        domain_dir = NEW_CORPUS / domain

        domain_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for entry in entries:

            assert len(entry) == 11

            document_counter += 1

            title = entry[0]

            document_id = (
                f"V2_DOC_{document_counter:03d}"
            )

            assert document_id not in old_document_ids

            fields = list(
                zip(entry[1::2], entry[2::2])
            )

            assert len(fields) == 5

            filename = (
                f"v2_{document_counter:03d}_"
                f"{slug(title)}.txt"
            )

            path = domain_dir / filename

            paragraphs = [
                "SYNTHETIC INSTITUTIONAL KNOWLEDGE BASE",
                f"Document ID: {document_id}",
                f"Domain: {domain}",
                f"Policy title: {title}",
                "",
                "Scope",
                (
                    f"This document describes the "
                    f"{title.lower()} procedure "
                    f"within the fictional institution."
                ),
                "",
                "Policy provisions"
            ]

            for index, (field, value) in enumerate(
                fields, start=1
            ):

                paragraphs.append(
                    f"{index}. {field}: {value}."
                )

            paragraphs.extend([
                "",
                "Administrative guidance",
                (
                    "Students should consult the "
                    "designated institutional office "
                    "for clarification of this policy."
                ),
                (
                    "This document is synthetic and "
                    "is intended for controlled "
                    "academic research."
                )
            ])

            text = "\n".join(paragraphs)

            if path.exists():

                assert path.read_text(
                    encoding="utf-8"
                ) == text, f"File differs: {path}"

            else:

                path.write_text(
                    text,
                    encoding="utf-8"
                )

            manifest.append({
                "document_id": document_id,
                "domain": domain,
                "title": title,
                "filename": filename,
                "path": path.relative_to(
                    ROOT
                ).as_posix(),
                "sha256": sha256(path),
                "source": "synthetic_institutional_v2",
                "qa_count": 5,
                "review_status": "pending"
            })

            for index, (field, value) in enumerate(
                fields, start=1
            ):

                qa_id = (
                    f"V2_QA_{document_counter:03d}"
                    f"_{index:02d}"
                )

                question = (
                    f"According to the {title} policy, "
                    f"what is the {field.lower()}?"
                )

                new_qa.append({
                    "qa_id": qa_id,
                    "document_id": document_id,
                    "domain": domain,
                    "question": question,
                    "expected_answer": value,
                    "clean_context": text,
                    "source": "synthetic_institutional_v2",
                    "language": "en",
                    "answerable": True,
                    "review_status": "pending"
                })

    assert document_counter == 50

    assert len(new_qa) == 250

    assert len({
        r["qa_id"] for r in new_qa
    }) == 250

    assert all(
        r["expected_answer"] in r["clean_context"]
        for r in new_qa
    )

    expanded_documents = list(
        NEW_CORPUS.rglob("*.txt")
    )

    assert len(expanded_documents) == 100, (
        f"Expected 100 documents, "
        f"found {len(expanded_documents)}"
    )

    for relative, original_hash in (
        original_hashes.items()
    ):

        assert sha256(
            OLD_CORPUS / relative
        ) == original_hash

        assert sha256(
            NEW_CORPUS / relative
        ) == original_hash

    combined_qa = []

    for record in old_benign:

        combined_qa.append({
            "qa_id": record["qa_id"],
            "document_id": record["document_id"],
            "domain": record["domain"],
            "question": record["question"],
            "expected_answer": record["expected_answer"],
            "clean_context": record["clean_context"],
            "source": "synthetic_institutional_v1",
            "language": "en",
            "answerable": True,
            "review_status": "pending"
        })

    combined_qa.extend(new_qa)

    assert len(combined_qa) == 500

    assert len({
        r["qa_id"] for r in combined_qa
    }) == 500

    save_jsonl(
        PROCESSED / "new_qa_pairs.jsonl",
        new_qa
    )

    save_jsonl(
        PROCESSED / "expanded_qa_pairs.jsonl",
        combined_qa
    )

    save_jsonl(
        METADATA / "new_corpus_manifest.jsonl",
        manifest
    )

    domain_counts = Counter(
        path.relative_to(NEW_CORPUS).parts[0]
        for path in expanded_documents
    )

    report = {
        "original_documents": 50,
        "new_documents": document_counter,
        "total_documents": len(expanded_documents),
        "original_qa_pairs": len(old_benign),
        "new_qa_pairs": len(new_qa),
        "total_qa_pairs": len(combined_qa),
        "document_domain_distribution": dict(
            domain_counts
        ),
        "qa_domain_distribution": dict(
            Counter(
                r["domain"]
                for r in combined_qa
            )
        ),
        "original_documents_preserved": True,
        "original_dataset_modified": False,
        "new_answers_grounded": True,
        "manual_policy_review_complete": False,
        "cross_document_consistency_verified": False
    }

    report_path = (
        METADATA / "expanded_corpus_statistics.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("=" * 60)
    print("EXPANDED INSTITUTIONAL CORPUS")
    print("=" * 60)

    print(
        json.dumps(report, indent=2)
    )

    print("\nCORPUS EXPANSION COMPLETED")


if __name__ == "__main__":
    main()
