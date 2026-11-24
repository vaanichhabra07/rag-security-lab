# RAG Security Lab

## Evaluating the Generalization of Prompt-Injection Defenses in Retrieval-Augmented Generation Systems

A B.Tech final-year major project focused on evaluating the effectiveness and generalization of prompt-injection defenses in Retrieval-Augmented Generation (RAG) systems.

## Project Objectives

- Develop a baseline RAG pipeline.
- Construct a labelled benchmark dataset covering five prompt-injection attack families.
- Implement five modular defense mechanisms.
- Evaluate defense effectiveness against seen and unseen attacks.
- Conduct ablation studies and cross-model comparisons across at least two LLMs.
- Perform statistical significance testing and security–utility trade-off analysis.

## Attack Categories

- Direct Injection
- Instruction Manipulation
- Obfuscation
- Indirect Injection
- Multi-stage Attacks

## Defense Configurations

- D0: No Defense
- D1: Rule-Based Detection
- D2: ML/LLM-Based Detection
- D3: Context Isolation
- D4: Output Validation
- D5: Combined Defense

## Dataset Strategy

The benchmark integrates publicly available BIPIA data, synthetic institutional documents, benign question-answer pairs and controlled supplementary attack samples.

## Evaluation Metrics

- Attack Success Rate (ASR)
- Detection Recall
- False Positive Rate (FPR)
- Legitimate-Task Utility
- Response Latency
- Token and API Cost
- Generalization Gap

## Project Status

Under active development — dataset construction phase.

## Ethical Scope

All experiments are conducted in a controlled research environment using public or synthetic data. No real production systems or private user data are targeted.