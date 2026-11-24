# RAG Security Lab Dataset

## Dataset Information

- Version: 1.0-rc1
- Documentation date: 2026-11-25
- Status: Structurally validated research candidate
- Language: English
- Domain: Higher education
- Total samples: 1500
- Institutional documents: 50
- Unique QA pairs: 250

## Research Objective

Evaluating the Generalization of Prompt-Injection Defenses in Retrieval-Augmented Generation Systems.

The dataset supports controlled experiments on prompt-injection detection, prevention, response validation, and generalization to held-out templates.

## Data Sources

1. Synthetic institutional knowledge documents.
2. Custom controlled adversarial fixtures.
3. Microsoft BIPIA benchmark templates.

BIPIA repository:
https://github.com/microsoft/BIPIA

BIPIA research paper:
https://arxiv.org/abs/2312.14197

## Institutional Domains

- academics: 300 samples
- administration: 300 samples
- attendance: 300 samples
- examinations: 300 samples
- placements: 300 samples

## Label Distribution

| Category | Samples |
|---|---:|
| adversarial | 1250 |
| benign | 250 |

## Attack Family Distribution

| Category | Samples |
|---|---:|
| direct_injection | 250 |
| indirect_injection | 250 |
| instruction_manipulation | 250 |
| multi_stage | 250 |
| none | 250 |
| obfuscation | 250 |

## Source Distribution

| Category | Samples |
|---|---:|
| BIPIA_adapted | 13 |
| BIPIA_derived | 287 |
| custom | 950 |
| synthetic_institutional | 250 |

## Dataset Splits

| Category | Samples |
|---|---:|
| test_benign | 100 |
| test_seen | 250 |
| test_unseen | 250 |
| train | 600 |
| validation | 300 |

## Attack Success Criteria

| Category | Samples |
|---|---:|
| exact_match | 950 |
| marker_present | 5 |
| semantic_review | 295 |

## Dataset Splitting Methodology

- Document-level train/validation/test separation.
- Training and validation use development templates.
- Seen testing uses familiar template IDs.
- Unseen testing uses held-out template IDs.
- Seen and unseen tests share QA-family pairs.
- Benign testing measures false positives and utility.

## Planned Defense Configurations

- D0: Undefended baseline RAG.
- D1: Rule-based detection.
- D2: Machine-learning-based detection.
- D3: Context isolation.
- D4: Output validation.
- D5: Combined defense.

## Planned Evaluation Metrics

- Attack Success Rate (ASR)
- Detection Recall
- False Positive Rate (FPR)
- Answer Utility
- Average Latency
- Token Consumption
- API Cost

## Structural Validation

- Passed checks: 38
- Failed checks: 0
- Structural validation: True

## Validation Warnings

- 50 paired tests have different attack sources.
- 295 samples require validated semantic scoring.
- Exact payload checks do not establish semantic independence of attack templates.
- Custom attack diversity and family definitions require qualitative review.

## Known Research Limitations

- Institutional documents are synthetic.
- Custom samples reuse controlled template patterns.
- Held-out wording does not necessarily establish semantic novelty.
- BIPIA output obfuscation differs from input obfuscation.
- Some BIPIA templates are adapted for safe testing.
- Semantic attack scoring requires validation.
- Simulated conversation histories are not equivalent to live multi-turn interactions.
- Source differences may confound seen/unseen comparisons.
- Structural validation does not establish complete research validity.

## Ethical Considerations

This dataset is intended for controlled defensive AI security research. Institutional records are synthetic. Experiments must remain isolated from production systems and external actions.

## Reproducibility

- Dataset planning seed: 42.
- Generation scripts are maintained in scripts/.
- Master dataset is stored in data/processed/.
- Experimental partitions are stored in data/splits/.
- Metadata is stored in data/metadata/.

## Licensing

Original BIPIA material remains subject to its upstream license. Redistribution permissions must be verified before publishing the combined dataset.

## Intended Use

Academic research, defensive RAG evaluation, prompt-injection detection, and controlled security benchmarking.
