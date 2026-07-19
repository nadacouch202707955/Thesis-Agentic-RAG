# Agentic RAG for Personalised Academic Advising

An Agentic Retrieval-Augmented Generation (RAG) framework for personalised academic advising, built on Microsoft Azure AI Foundry, developed as part of an MSc Artificial Intelligence thesis at Bahrain Polytechnic.

## Project Overview

This research designs, implements, and evaluates an Agentic RAG system that grounds academic advising responses in institutional knowledge (regulations, programme catalogues, student handbooks, academic calendars) while incorporating specialised AI agents, hybrid retrieval, and Human-in-the-Loop (HITL) governance. The system is benchmarked against a conventional (Basic) RAG baseline using RAGAS metrics and statistical significance testing.

**Methodology:** Design Science Research (DSR)
**Platform:** Microsoft Azure AI Foundry (Azure AI Search, Azure OpenAI, Azure SQL Database)

## Research Objectives

| # | Objective | Status |
|---|---|---|
| O1 | Compare Agentic RAG vs. Basic RAG performance using RAGAS metrics on a 50-question benchmark | In progress — Basic RAG validated; benchmark complete; Agentic RAG build next |
| O2 | Determine statistical significance of performance differences via paired t-test | Awaiting RQ1 RAGAS results |
| O3 | Evaluate chunking strategies, embedding configuration, and retrieval settings using Precision@5 | 150 retrieval searches complete across kb-256, kb-512, kb-1024; scoring in progress |
| O4 | Identify the optimal retrieval configuration (highest Precision@5) | Retrieval results saved; winner selection pending scorer output |

## Repository Structure
├── ingestion_pipeline.py           # PDF ingestion, chunking, and indexing pipeline
├── basic_rag.py                    # Basic RAG baseline (hybrid BM25 + vector retrieval, GPT-5-mini)
├── retrieval_eval_rq2.py           # RQ2 retrieval evaluation across kb-256/512/1024
├── p5_scorer_rq2.py                # Precision@5 scoring script
├── resolve_ground_truth_pages.py   # Ground-truth page resolution utility
├── benchmark_50_questions.json     # Draft 50-question benchmark
├── benchmark_50_questions_verified.json  # Final verified benchmark (ground truth + source + page ref)
├── retrieval_results_rq2.json      # Results of 150 hybrid retrieval searches
├── ingestion_summary.json          # Summary of document ingestion/chunking run
├── requirements_ingestion.txt      # Python dependencies for ingestion pipeline
├── chapter4_evidence_log.md        # Evidence log for thesis Chapter 4 (Implementation)
├── limitations_paragraph_template.md
└── .gitignore
## Knowledge Base

- **Source:** 73 institutional PDF documents (academic regulations, programme catalogues, student handbooks, academic calendars)
- **Storage:** Azure Blob Storage (`academic-advising-docs` container)
- **Indexing:** Azure AI Search with hybrid BM25 + dense vector retrieval
- **Embedding model:** `text-embedding-3-small` (fixed across all configurations)
- **Chunking configurations tested:**
  | Index | Chunks | Overlap |
  |---|---|---|
  | kb-256 | 1,271 | 51 tokens |
  | kb-512 | 689 | 102 tokens |
  | kb-1024 | 438 | 205 tokens |

## Evaluation Plan

- **Benchmark:** 50 verified questions (20 academic policy, 20 course eligibility, 10 deadline-related), each with ground-truth answer, source document, and page reference
- **Retrieval evaluation:** Retrieval Precision@5 across three chunking configurations (150 total searches)
- **Response quality:** RAGAS Faithfulness and Answer Relevance (Basic RAG vs. Agentic RAG, 4 systems × 50 questions)
- **Statistical test:** Paired t-test on RAGAS scores
- **User evaluation:** ≥20 student participants, 5 standardised advising tasks each, System Usability Scale (SUS) questionnaires, 2–3 semi-structured advisor interviews

## Tech Stack

- **LLM / Generation:** Azure OpenAI (GPT-5-mini for Basic RAG baseline; GPT-4o for RAGAS evaluation)
- **Retrieval:** Azure AI Search (hybrid BM25 + dense vector)
- **Storage:** Azure Blob Storage, Azure SQL Database (student profile data)
- **Ingestion:** PyMuPDF, LangChain `RecursiveCharacterTextSplitter`, tiktoken
- **Orchestration (planned):** Multi-agent architecture — Orchestrator, Retrieval, Profile, Validator, Notification agents
- **HITL escalation (planned):** Azure Logic Apps

## Current Status




## Author

Nada — MSc Artificial Intelligence, Bahrain Polytechnic
Supervisor: Faustino Reyes



