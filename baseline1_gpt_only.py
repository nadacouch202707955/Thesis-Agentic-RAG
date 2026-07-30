"""
baseline1_gpt_only.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

Baseline 1 — GPT-4o Alone (No Retrieval, No Agents)
====================================================
Chapter 3, Table 3.7 — Condition 1

This baseline represents the simplest possible AI advising system:
a plain GPT-4o call with a system prompt but NO document retrieval
and NO agent coordination. It answers entirely from model weights.

Used in RQ1 to establish the performance floor — how well does
GPT-4o perform on academic advising questions without any
institutional knowledge base?

Usage (interactive):
    py baseline1_gpt_only.py

Usage (batch evaluation — called by ragas_eval_rq1.py):
    from baseline1_gpt_only import run_gpt_only_query
"""

import os
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY  = os.getenv("AZURE_OPENAI_API_KEY")
CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini")

# ── System prompt (identical to Basic RAG and Agentic RAG) ─────────
# Keeping the system prompt constant isolates the effect of retrieval
# and agent architecture — the only variable across the 4 systems.
SYSTEM_PROMPT = """You are a professional academic advising assistant for a higher education institution.
Answer the student's question as accurately as possible based on your knowledge of academic policies and procedures.

IMPORTANT: If you are not certain of the answer, say so clearly rather than guessing.
Always be professional, accurate, and helpful."""


def run_gpt_only_query(query: str, verbose: bool = True) -> dict:
    """
    Runs a plain GPT query with no retrieval and no agents.
    Returns a dict compatible with ragas_eval_rq1.py.

    Parameters:
        query   — student academic advising question
        verbose — print output to terminal

    Returns:
        {
          "query":           str,
          "answer":          str,
          "contexts":        [],      # empty — no retrieval
          "ground_truth":    "",      # filled by eval script
          "system":          "baseline1_gpt_only",
          "latency_seconds": float,
        }
    """
    start = time.time()

    client = AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version="2024-02-01"
    )

    if verbose:
        print(f"\n{'─'*50}")
        print(f"[Baseline 1 — GPT-4o Only] Query: {query[:70]}...")

    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": query}
        ],
        max_completion_tokens=1200
    )

    answer = response.choices[0].message.content or ""
    elapsed = round(time.time() - start, 2)

    if verbose:
        print(f"[Baseline 1] Response ({elapsed}s): {answer[:150]}...")

    return {
        "query":           query,
        "answer":          answer,
        "contexts":        [],      # no retrieval — empty context list
        "ground_truth":    "",      # filled by ragas_eval_rq1.py
        "system":          "baseline1_gpt_only",
        "latency_seconds": elapsed,
    }


def run_chat():
    print("=" * 50)
    print("Baseline 1 — GPT-4o Alone (No Retrieval)")
    print("Chapter 3 Table 3.7 — Condition 1")
    print(f"Model: {CHAT_DEPLOYMENT} | No knowledge base | No agents")
    print("=" * 50)
    print("Type your question. Type 'quit' to exit.\n")

    while True:
        query = input("Student: ").strip()
        if not query:
            continue
        if query.lower() in ["quit", "exit", "q"]:
            print("Session ended.")
            break

        result = run_gpt_only_query(query, verbose=True)
        print(f"\nAdvisor: {result['answer']}")
        print(f"[Latency: {result['latency_seconds']}s | No retrieval | No agents]\n")


if __name__ == "__main__":
    run_chat()
