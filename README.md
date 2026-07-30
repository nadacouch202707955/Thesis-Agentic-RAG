# Agentic RAG for Personalised Academic Advising

An Agentic Retrieval-Augmented Generation (RAG) framework for personalised academic advising, built on Microsoft Azure AI Foundry, developed as part of an MSc Artificial Intelligence thesis at Bahrain Polytechnic.

## Project Overview

This research designs, implements, and evaluates an Agentic RAG system that grounds academic advising responses in institutional knowledge (regulations, programme catalogues, student handbooks, academic calendars) while incorporating specialised AI agents, hybrid retrieval, and Human-in-the-Loop (HITL) governance. The system is benchmarked against three baselines (GPT-4o alone, Conventional RAG, and Single-Agent RAG) using RAGAS metrics and statistical significance testing.

**Methodology:** Design Science Research (DSR)
**Platform:** Microsoft Azure AI Foundry (Azure AI Search, Azure OpenAI, Azure SQL Database)

## Research Questions

| # | Question | Status |
|---|---|---|
| RQ1 | Does Agentic RAG improve response Faithfulness and Answer Relevance compared to simpler baselines? | **Complete.** Evaluated via RAGAS (GPT-4o judge) across 4 systems × 50 questions. See `ragas_eval_rq1.py`, `statistical_analysis.py`, and results files below. |
| RQ2 | Which retrieval configuration (chunking strategy) achieves the highest retrieval precision? | **Complete.** kb-512 achieved the highest Precision@5 (0.544) across 150 retrieval searches. See `retrieval_eval_rq2.py`, `p5_scorer_rq2.py`. |

## Repository Structure

```
├── ingestion_pipeline.py            # PDF ingestion, chunking, and indexing pipeline
├── basic_rag.py                     # Conventional RAG baseline (hybrid BM25 + vector retrieval)
├── baseline1_gpt_only.py            # GPT-4o-only baseline (no retrieval)
├── baseline3_single_agent.py        # Single-agent RAG baseline
├── orchestrator_agent.py            # Proposed Agentic RAG system (5-agent pipeline:
│                                     #   Orchestrator, Retrieval, Profile, Validator, Notification)
├── agentic_rag.py                   # Evaluation wrapper for the Agentic RAG system
├── config_frozen.py                 # Frozen retrieval configuration (RQ2 winner: kb-512)
├── retrieval_eval_rq2.py            # RQ2 retrieval evaluation across kb-256/512/1024
├── p5_scorer_rq2.py                 # Precision@5 scoring script
├── resolve_ground_truth_pages.py    # Ground-truth page resolution utility
├── ragas_eval_rq1.py                # Formal RQ1 RAGAS evaluation (4 systems x 50 questions)
├── statistical_analysis.py          # Paired t-test + Cohen's d effect size analysis
├── demo_app.py                      # Streamlit live demonstration interface
├── benchmark_50_questions_verified.json  # Final verified benchmark (ground truth + source + page ref)
├── retrieval_results_rq2.json       # Results of 150 hybrid retrieval searches
├── ragas_results_all_systems.json   # Full RAGAS scoring results, all systems, all questions
├── ragas_summary.json               # Mean RAGAS scores per system (Table 5.1)
├── statistical_results.json         # H1/H2 paired t-test and Cohen's d results (Section 5.5)
├── escalations.json                 # Logged Human-in-the-Loop escalation records
├── ingestion_summary.json           # Summary of document ingestion/chunking run
├── requirements_ingestion.txt       # Python dependencies for ingestion pipeline
├── chapter4_evidence_log.md         # Evidence log for thesis Chapter 4 (Implementation)
└── .gitignore
```

## Knowledge Base

- **Source:** 73 institutional PDF documents (academic regulations, programme catalogues, student handbooks, academic calendars)
- **Storage:** Azure Blob Storage (`academic-advising-docs` container)
- **Indexing:** Azure AI Search with hybrid BM25 + dense vector retrieval
- **Embedding model:** `text-embedding-3-small` (fixed across all configurations)
- **Chunking configurations tested:**

| Index | Chunks | Overlap | Precision@5 |
|---|---|---|---|
| kb-256 | 1,271 | 51 tokens | — |
| **kb-512** | **689** | **102 tokens** | **0.544 (winner, frozen for RQ1)** |
| kb-1024 | 438 | 205 tokens | — |

## Evaluation Summary

- **Benchmark:** 50 verified questions (academic policy, course eligibility, deadline-related), each with ground-truth answer, source document, and page reference
- **Retrieval evaluation (RQ2):** Precision@5 across three chunking configurations (150 total searches) — kb-512 selected and frozen
- **Response quality (RQ1):** RAGAS Faithfulness and Answer Relevance across 4 systems x 50 questions, scored using GPT-4o as an LLM judge
- **Statistical test:** Paired t-test (H1: Faithfulness, H2: Answer Relevance) with Cohen's d effect size, comparing Conventional RAG vs. Agentic RAG
- **Key finding:** Aggregate RAGAS scores initially appeared lower for Agentic RAG due to its Human-in-the-Loop escalation behaviour (24% of questions escalated, scored as maximally unfaithful by RAGAS design). A matched comparison on the same answered-question subset showed Agentic RAG's Faithfulness and Answer Relevance to be statistically comparable to baselines, while Agentic RAG behaved more conservatively specifically on the harder questions where baselines scored substantially lower. See thesis Chapter 5 (§5.5) and Chapter 6 (§6.1–6.2) for full analysis.

## Tech Stack

- **LLM / Generation:** Azure OpenAI GPT-4o (used throughout the formal evaluation; GPT-5-mini used only during early development/smoke testing)
- **Retrieval:** Azure AI Search (hybrid BM25 + dense vector)
- **Storage:** Azure Blob Storage, Azure SQL Database (student profile data, 4,424 students)
- **Ingestion:** PyMuPDF, LangChain `RecursiveCharacterTextSplitter`, tiktoken
- **Evaluation:** RAGAS (v0.1.9, run in an isolated Python 3.11 environment due to a dependency incompatibility with Python 3.14), scipy (paired t-test, Cohen's d)
- **Orchestration:** Custom multi-agent architecture — Orchestrator, Retrieval, Profile, Validator, Notification agents (no external agent framework)
- **HITL escalation:** Logged to `escalations.json`; optional Azure Logic Apps webhook trigger supported via `AZURE_LOGIC_APPS_URL`
- **Demo interface:** Streamlit (`demo_app.py`) — run with `streamlit run demo_app.py`

## Reproducibility

1. Clone this repository and install dependencies: `pip install -r requirements_ingestion.txt`
2. Set up a `.env` file (not committed — see `.gitignore`) with Azure OpenAI, Azure AI Search, Azure Storage, and Azure SQL credentials
3. Run `python ingestion_pipeline.py` to build the knowledge base index
4. Run `python retrieval_eval_rq2.py` and `python p5_scorer_rq2.py` to reproduce RQ2 results
5. For RQ1 evaluation, create a separate Python 3.11 virtual environment and install `ragas==0.1.9`, `langchain==0.1.20`, `langchain-core==0.1.52`, `langchain-openai==0.1.7`, `langchain-community==0.0.38` (see thesis Chapter 4 for full environment notes), then run `python ragas_eval_rq1.py` followed by `python statistical_analysis.py`
6. To try the live demo: `pip install streamlit` then `streamlit run demo_app.py`

## Author

Nada Ali Yaqoob — MSc Artificial Intelligence, Bahrain Polytechnic (Student ID: 202507955)
Supervisor: Dr. Faustino Reyes

## About

An Agentic Retrieval-Augmented Generation Framework for Personalised Academic Advising Using Microsoft Azure AI Foundry at Higher Education Institutions.
