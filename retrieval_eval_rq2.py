"""
retrieval_eval_rq2.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

Purpose
-------
RQ2 needs to know: for each of the 50 benchmark questions, which chunks
come back from kb-256 vs kb-512 vs kb-1024? This script answers that —
retrieval only, no GPT call, so it's fast and free of generation cost.

It deliberately REUSES embed_query() and hybrid_search() from basic_rag.py
instead of rewriting them, so RQ2 retrieval and Basic RAG retrieval are
guaranteed to behave identically — the only thing that changes across runs
is which index is queried.

Run: python retrieval_eval_rq2.py
Output: retrieval_results_rq2.json
    { "kb-256": { "Q001": [ {source_document, source_page, content, score}, ... 5 chunks ] }, ... }
"""

import os
import json
import time
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# Reuse the already-working functions instead of duplicating logic
from basic_rag import embed_query, hybrid_search

load_dotenv()

SEARCH_ENDPOINT  = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
OPENAI_ENDPOINT  = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY   = os.getenv("AZURE_OPENAI_API_KEY")

KB_INDEXES = ["kb-256", "kb-512", "kb-1024"]

BENCHMARK_FILE = "benchmark_50_questions_verified.json"   # verified page numbers —
                                                          # produced by resolve_ground_truth_pages.py   # will be replaced with the
                                                   # page-verified version once
                                                   # resolve_ground_truth_pages.py has run
OUTPUT_FILE = "retrieval_results_rq2.json"


def get_search_client(index_name: str) -> SearchClient:
    return SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=index_name,
        credential=AzureKeyCredential(SEARCH_ADMIN_KEY),
    )


def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version="2024-02-01",
    )


def run():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    openai_client = get_openai_client()
    all_results = {}

    for kb_name in KB_INDEXES:
        print(f"\n{'='*50}\nQuerying {kb_name}\n{'='*50}")
        search_client = get_search_client(kb_name)
        kb_results = {}

        for i, q in enumerate(benchmark["questions"], start=1):
            query = q["question"]
            print(f"[{i}/50] {q['id']}: {query[:55]}...")

            query_vector = embed_query(openai_client, query)
            chunks = hybrid_search(search_client, query, query_vector)

            kb_results[q["id"]] = [
                {
                    "source_document": c["source_document"],
                    "source_page": c["source_page"],
                    "content_snippet": c["content"][:200],
                    "score": c["score"],
                }
                for c in chunks
            ]

            time.sleep(0.2)  # gentle on rate limits — 50 questions x 3 indexes = 150 calls

        all_results[kb_name] = kb_results

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Retrieval results for all 3 indexes saved to {OUTPUT_FILE}")
    print("Next step: run p5_scorer_rq2.py to compute Precision@5.")


if __name__ == "__main__":
    run()
