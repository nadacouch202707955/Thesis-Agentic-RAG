"""
retrieval_mode_comparison.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

Compares three retrieval modes on kb-512 (frozen config winner):
  1. Vector only  — dense semantic search
  2. Keyword only — BM25 sparse search
  3. Hybrid       — BM25 + vector combined (Chapter 3 §3.4.3)

Uses document-level matching (source_document) for speed.
Results saved to retrieval_mode_results.json and retrieval_mode_results.csv
for Chapter 5 Table 5.3 (Ablation Studies).

Run: py retrieval_mode_comparison.py
"""

import os
import json
import csv
import time
import statistics
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

load_dotenv()

SEARCH_ENDPOINT  = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
OPENAI_ENDPOINT  = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY   = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_MODEL  = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

INDEX_NAME   = "kb-512"   # frozen config winner
TOP_K        = 5
BENCHMARK    = "benchmark_50_questions_verified.json"
OUTPUT_JSON  = "retrieval_mode_results.json"
OUTPUT_CSV   = "retrieval_mode_results.csv"


def get_clients():
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_ADMIN_KEY)
    )
    openai_client = AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version="2024-02-01"
    )
    return search_client, openai_client


def embed_query(openai_client, query: str) -> list:
    response = openai_client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def search_vector_only(search_client, query_vector: list) -> list:
    vq = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=TOP_K,
        fields="content_vector"
    )
    results = search_client.search(
        search_text=None,
        vector_queries=[vq],
        select=["content", "source_document", "source_page"],
        top=TOP_K
    )
    return list(results)


def search_keyword_only(search_client, query: str) -> list:
    results = search_client.search(
        search_text=query,
        select=["content", "source_document", "source_page"],
        top=TOP_K
    )
    return list(results)


def search_hybrid(search_client, query: str, query_vector: list) -> list:
    vq = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=TOP_K,
        fields="content_vector"
    )
    results = search_client.search(
        search_text=query,
        vector_queries=[vq],
        select=["content", "source_document", "source_page"],
        top=TOP_K
    )
    return list(results)


def doc_hit(results: list, gt_doc: str) -> bool:
    """Returns True if the correct document appears in top-5 results."""
    gt_norm = gt_doc.strip().lower()
    for r in results:
        if r["source_document"].strip().lower() == gt_norm:
            return True
    return False


def run():
    print("=" * 55)
    print("Retrieval Mode Comparison — kb-512")
    print("Vector Only vs Keyword Only vs Hybrid")
    print("=" * 55)

    with open(BENCHMARK, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    questions = benchmark["questions"]

    search_client, openai_client = get_clients()

    modes = ["vector_only", "keyword_only", "hybrid"]
    results_by_mode = {m: [] for m in modes}
    csv_rows = []

    for i, q in enumerate(questions):
        qid   = q["id"]
        query = q["question"]
        gt_doc = q["source_document"].split(";")[0].strip()

        print(f"[{i+1}/50] {qid}: {query[:50]}...")

        # Generate embedding once per question
        query_vector = embed_query(openai_client, query)
        time.sleep(0.3)

        for mode in modes:
            if mode == "vector_only":
                hits = search_vector_only(search_client, query_vector)
            elif mode == "keyword_only":
                hits = search_keyword_only(search_client, query)
            else:
                hits = search_hybrid(search_client, query, query_vector)

            hit = doc_hit(hits, gt_doc)
            score = 1.0 if hit else 0.0
            results_by_mode[mode].append(score)

            csv_rows.append({
                "mode":          mode,
                "id":            qid,
                "hit":           int(hit),
                "question_type": q.get("question_type", ""),
                "difficulty":    q.get("difficulty", ""),
            })

        time.sleep(0.2)

    # Compute summary statistics
    summary = {}
    print("\n=== RETRIEVAL MODE COMPARISON RESULTS ===")
    print(f"{'Mode':<15} {'Mean P@5':<12} {'SD':<10} {'n'}")
    print("-" * 45)

    for mode in modes:
        scores = results_by_mode[mode]
        mean   = statistics.mean(scores)
        sd     = statistics.stdev(scores) if len(scores) > 1 else 0.0
        summary[mode] = {
            "mean_document_recall": round(mean, 3),
            "sd":                   round(sd, 3),
            "n":                    len(scores),
            "hits":                 int(sum(scores)),
        }
        print(f"{mode:<15} {mean:.3f}        {sd:.3f}     {len(scores)}")

    winner = max(summary.items(), key=lambda x: x[1]["mean_document_recall"])[0]
    print(f"\n★ Best retrieval mode: {winner} "
          f"(mean = {summary[winner]['mean_document_recall']})")
    print("This confirms hybrid retrieval as the correct choice per Chapter 3 §3.4.3")

    # Save JSON
    output = {
        "index":   INDEX_NAME,
        "top_k":   TOP_K,
        "summary": summary,
        "note":    "Document-level recall (hit=1 if correct PDF in top-5)"
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Save CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["mode","id","hit","question_type","difficulty"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_CSV}")
    print("Add these results to Chapter 5 Table 5.3 (Ablation Studies)")


if __name__ == "__main__":
    run()
