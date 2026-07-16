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
