
import json
from pathlib import Path
from collections import Counter
from datetime import date

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
METADATA = ROOT / "data" / "metadata"


def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():

    records = load_jsonl(
        PROCESSED / "master_dataset.jsonl"
    )

    with (
        METADATA / "validation_report.json"
    ).open("r", encoding="utf-8") as f:
        validation = json.load(f)

    assert len(records) == 1500

    assert validation["structural_validation_passed"], (
        "Dataset must pass structural validation."
    )

    labels = Counter(r["label"] for r in records)
    families = Counter(r["attack_family"] for r in records)
    sources = Counter(r["source"] for r in records)
    splits = Counter(r["split"] for r in records)
    domains = Counter(r["domain"] for r in records)

    criteria = Counter(
        r["success_criteria"]["type"]
        for r in records
        if r["label"] == "adversarial"
    )

    documents = len({
        r["document_id"] for r in records
    })

    questions = len({
        r["qa_id"] for r in records
    })

    lines = [
        "# RAG Security Lab Dataset",
        "",
        "## Dataset Information",
        "",
        "- Version: 1.0-rc1",
        f"- Documentation date: {date.today()}",
        "- Status: Structurally validated research candidate",
        "- Language: English",
        "- Domain: Higher education",
        f"- Total samples: {len(records)}",
        f"- Institutional documents: {documents}",
        f"- Unique QA pairs: {questions}",
        "",
        "## Research Objective",
        "",
        "Evaluating the Generalization of Prompt-Injection "
        "Defenses in Retrieval-Augmented Generation Systems.",
        "",
        "The dataset supports controlled experiments on "
        "prompt-injection detection, prevention, response "
        "validation, and generalization to held-out templates.",
        "",
        "## Data Sources",
        "",
        "1. Synthetic institutional knowledge documents.",
        "2. Custom controlled adversarial fixtures.",
        "3. Microsoft BIPIA benchmark templates.",
        "",
        "BIPIA repository:",
        "https://github.com/microsoft/BIPIA",
        "",
        "BIPIA research paper:",
        "https://arxiv.org/abs/2312.14197",
        "",
        "## Institutional Domains",
        ""
    ]

    def add_table(title, counts):

        lines.extend([
            f"## {title}",
            "",
            "| Category | Samples |",
            "|---|---:|"
        ])

        for name, count in sorted(counts.items()):
            lines.append(f"| {name} | {count} |")

        lines.append("")

    for domain, count in sorted(domains.items()):
        lines.append(f"- {domain}: {count} samples")

    lines.append("")

    add_table("Label Distribution", labels)
    add_table("Attack Family Distribution", families)
    add_table("Source Distribution", sources)
    add_table("Dataset Splits", splits)
    add_table("Attack Success Criteria", criteria)

    lines.extend([
        "## Dataset Splitting Methodology",
        "",
        "- Document-level train/validation/test separation.",
        "- Training and validation use development templates.",
        "- Seen testing uses familiar template IDs.",
        "- Unseen testing uses held-out template IDs.",
        "- Seen and unseen tests share QA-family pairs.",
        "- Benign testing measures false positives and utility.",
        "",
        "## Planned Defense Configurations",
        "",
        "- D0: Undefended baseline RAG.",
        "- D1: Rule-based detection.",
        "- D2: Machine-learning-based detection.",
        "- D3: Context isolation.",
        "- D4: Output validation.",
        "- D5: Combined defense.",
        "",
        "## Planned Evaluation Metrics",
        "",
        "- Attack Success Rate (ASR)",
        "- Detection Recall",
        "- False Positive Rate (FPR)",
        "- Answer Utility",
        "- Average Latency",
        "- Token Consumption",
        "- API Cost",
        "",
        "## Structural Validation",
        "",
        f"- Passed checks: {validation['checks_passed']}",
        f"- Failed checks: {validation['checks_failed']}",
        f"- Structural validation: "
        f"{validation['structural_validation_passed']}",
        "",
        "## Validation Warnings",
        ""
    ])

    for warning in validation.get("warnings", []):
        lines.append(f"- {warning}")

    lines.extend([
        "",
        "## Known Research Limitations",
        "",
        "- Institutional documents are synthetic.",
        "- Custom samples reuse controlled template patterns.",
        "- Held-out wording does not necessarily establish "
        "semantic novelty.",
        "- BIPIA output obfuscation differs from input "
        "obfuscation.",
        "- Some BIPIA templates are adapted for safe testing.",
        "- Semantic attack scoring requires validation.",
        "- Simulated conversation histories are not equivalent "
        "to live multi-turn interactions.",
        "- Source differences may confound seen/unseen "
        "comparisons.",
        "- Structural validation does not establish complete "
        "research validity.",
        "",
        "## Ethical Considerations",
        "",
        "This dataset is intended for controlled defensive "
        "AI security research. Institutional records are "
        "synthetic. Experiments must remain isolated from "
        "production systems and external actions.",
        "",
        "## Reproducibility",
        "",
        "- Dataset planning seed: 42.",
        "- Generation scripts are maintained in scripts/.",
        "- Master dataset is stored in data/processed/.",
        "- Experimental partitions are stored in data/splits/.",
        "- Metadata is stored in data/metadata/.",
        "",
        "## Licensing",
        "",
        "Original BIPIA material remains subject to its "
        "upstream license. Redistribution permissions must "
        "be verified before publishing the combined dataset.",
        "",
        "## Intended Use",
        "",
        "Academic research, defensive RAG evaluation, "
        "prompt-injection detection, and controlled "
        "security benchmarking.",
        ""
    ])

    output = METADATA / "dataset_card.md"

    output.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("=" * 55)
    print("DATASET CARD GENERATED SUCCESSFULLY")
    print("=" * 55)

    print(f"Total samples: {len(records)}")
    print(f"Documents: {documents}")
    print(f"QA pairs: {questions}")
    print(f"Dataset version: 1.0-rc1")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
