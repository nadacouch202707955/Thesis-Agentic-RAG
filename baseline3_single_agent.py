"""
baseline3_single_agent.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

Baseline 3 — Single-Agent RAG
==============================
Chapter 3, Table 3.7 — Condition 3

This baseline uses ONE agent that retrieves documents and generates
a response — but has NO student profile lookup (no Azure SQL),
NO confidence validation, and NO HITL escalation.

It sits between Conventional RAG (Baseline 2) and the full
Agentic RAG system (Proposed System), isolating the contribution
of the multi-agent architecture over a single-agent approach.

Architecture:
    Query → Single Agent → Retrieval (kb-512 frozen) → GPT → Response

No personalisation, no validation, no escalation path.

Usage (interactive):
    py baseline3_single_agent.py

Usage (batch evaluation — called by ragas_eval_rq1.py):
    from baseline3_single_agent import run_single_agent_query
"""

import os
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

# Import retrieval functions from basic_rag.py (frozen config)
from basic_rag import embed_query, hybrid_search, get_clients
from config_frozen import get_frozen_config

load_dotenv()

FROZEN          = get_frozen_config()
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")
CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini")

# ── System prompt (identical across all 4 systems) ─────────────────
SYSTEM_PROMPT = """You are a professional academic advising assistant for a higher education institution.
You have access to official institutional policy documents.

STRICT RULES:
1. Answer ONLY using the retrieved document chunks provided below.
2. ALWAYS cite your source: [Source: <document name>, Page <number>]
3. If the retrieved context is insufficient, respond:
   "I don't have sufficient information in the available documents to answer this.
   Please consult your academic advisor directly."
4. NEVER guess, invent, or assume policy details not present in the retrieved chunks.
5. Be concise, accurate, and professional."""


def run_single_agent_query(query: str, verbose: bool = True) -> dict:
    """
    Runs a single-agent RAG query.
    One agent handles retrieval + response generation.
    No student profile, no validator, no HITL.
    Returns a dict compatible with ragas_eval_rq1.py.

    Parameters:
        query   — student academic advising question
        verbose — print output to terminal

    Returns:
        {
          "query":           str,
          "answer":          str,
          "contexts":        list[str],  # retrieved chunk texts
          "ground_truth":    "",         # filled by eval script
          "system":          "baseline3_single_agent",
          "latency_seconds": float,
        }
    """
    start = time.time()

    search_client, openai_client = get_clients()

    if verbose:
        print(f"\n{'─'*50}")
        print(f"[Baseline 3 — Single Agent] Query: {query[:70]}...")

    # ── Step 1: Retrieve using frozen config ───────────────────────
    query_vector = embed_query(openai_client, query)
    chunks = hybrid_search(search_client, query, query_vector)

    if verbose:
        print(f"  [Retrieval] {len(chunks)} chunks from {FROZEN['index_name']}")
        for i, c in enumerate(chunks):
            print(f"    [{i+1}] {c['source_document']} p.{c['source_page']} "
                  f"(score: {c.get('score', 0):.3f})")

    # ── Step 2: Build prompt ───────────────────────────────────────
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Chunk {i+1}] Source: {chunk['source_document']}, "
            f"Page {chunk['source_page']}\n{chunk['content']}"
        )
    context_block = "\n\n".join(context_parts)

    user_message = f"""Retrieved document chunks:
{context_block}

Student question: {query}"""

    # ── Step 3: Generate response ──────────────────────────────────
    gen_client = AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version="2024-02-01"
    )

    response = gen_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        max_completion_tokens=1200
    )

    answer  = response.choices[0].message.content or ""
    elapsed = round(time.time() - start, 2)

    if verbose:
        print(f"  [Single Agent] Response approved — no validation step")
        print(f"[Baseline 3] Complete in {elapsed}s")

    # Extract context texts for RAGAS evaluation
    contexts = [c["content"] for c in chunks]

    return {
        "query":           query,
        "answer":          answer,
        "contexts":        contexts,
        "ground_truth":    "",      # filled by ragas_eval_rq1.py
        "system":          "baseline3_single_agent",
        "latency_seconds": elapsed,
    }


def run_chat():
    print("=" * 50)
    print("Baseline 3 — Single-Agent RAG")
    print("Chapter 3 Table 3.7 — Condition 3")
    print(f"Index: {FROZEN['index_name']} | Model: {CHAT_DEPLOYMENT}")
    print("No student profile | No validator | No HITL")
    print("=" * 50)
    print("Type your question. Type 'quit' to exit.\n")

    while True:
        query = input("Student: ").strip()
        if not query:
            continue
        if query.lower() in ["quit", "exit", "q"]:
            print("Session ended.")
            break

        result = run_single_agent_query(query, verbose=True)
        print(f"\nAdvisor: {result['answer']}")
        print(f"[Latency: {result['latency_seconds']}s | "
              f"Single agent | No profile | No validation]\n")


if __name__ == "__main__":
    run_chat()
