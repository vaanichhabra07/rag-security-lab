
import json

from pathlib import Path
from dataclasses import dataclass


ROOT = Path(__file__).resolve().parents[2]

CORPUS_DIR = ROOT / "data" / "v2" / "corpus"

QA_FILE = (
    ROOT / "data" / "v2"
    / "processed" / "expanded_qa_pairs.jsonl"
)


@dataclass
class Document:

    document_id: str
    domain: str
    text: str
    source: str


def normalize_text(text):

    return " ".join(text.split())


def load_documents():

    if not QA_FILE.exists():

        raise FileNotFoundError(QA_FILE)

    corpus_files = sorted(
        CORPUS_DIR.rglob("*.txt")
    )

    if not corpus_files:

        raise ValueError(
            "No institutional documents found"
        )

    corpus_lookup = {}

    for path in corpus_files:

        text = path.read_text(
            encoding="utf-8-sig"
        ).strip()

        normalized = normalize_text(text)

        corpus_lookup.setdefault(
            normalized, []
        ).append(path)

    documents = {}

    with QA_FILE.open(
        encoding="utf-8-sig"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)

            document_id = record["document_id"]

            text = record["clean_context"].strip()

            if document_id in documents:

                previous = documents[document_id]

                if (
                    normalize_text(previous.text)
                    != normalize_text(text)
                ):

                    raise ValueError(
                        f"Inconsistent context: "
                        f"{document_id}"
                    )

                continue

            matches = corpus_lookup.get(
                normalize_text(text), []
            )

            if len(matches) != 1:

                raise ValueError(
                    f"Expected one matching corpus "
                    f"file for {document_id}; "
                    f"found {len(matches)}"
                )

            source = matches[0].relative_to(
                ROOT
            ).as_posix()

            documents[document_id] = Document(

                document_id=document_id,

                domain=record["domain"],

                text=text,

                source=source
            )

    return list(documents.values())
