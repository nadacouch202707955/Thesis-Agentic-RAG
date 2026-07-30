"""
ragas_eval_rq1.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

RQ1 Evaluation — RAGAS Faithfulness + Answer Relevance
=======================================================
Chapter 3, Table 3.7 — All Four Conditions

PURPOSE
-------
This script runs the formal RQ1 evaluation by:
1. Loading the 50-question verified benchmark dataset
2. Running all 4 systems on every question
3. Scoring each answer using RAGAS Faithfulness and Answer Relevance
4. Saving results for statistical analysis (statistical_analysis.py)

The four systems evaluated:
    System 1: GPT-4o alone          (baseline1_gpt_only.py)
    System 2: Conventional RAG      (basic_rag.py)
    System 3: Single-Agent RAG      (baseline3_single_agent.py)
    System 4: Agentic RAG           (agentic_rag.py)

RAGAS Metrics:
    Faithfulness      — does the answer stay within the retrieved context?
    Answer Relevance  — does the answer address the question asked?

Output files:
    ragas_results_all_systems.json  — full results per question per system
    ragas_results_all_systems.csv   — for thesis appendix and statistics
    ragas_summary.json              — mean scores per system (Table 5.1)

Run: py ragas_eval_rq1.py

NOTE: Switch AZURE_CHAT_DEPLOYMENT in .env to gpt-4o before running
      the formal evaluation. GPT-5-mini is for development only.
"""

import os
import json
import csv
import time
import statistics
from dotenv import load_dotenv

# Import all 4 systems
from baseline1_gpt_only     import run_gpt_only_query
from basic_rag              import run_basic_rag_query
from baseline3_single_agent import run_single_agent_query
from agentic_rag            import run_agentic_rag_query

load_dotenv()

BENCHMARK_FILE = "benchmark_50_questions_verified.json"
OUTPUT_JSON    = "ragas_results_all_systems.json"
OUTPUT_CSV     = "ragas_results_all_systems.csv"
SUMMARY_JSON   = "ragas_summary.json"
CALL_DELAY     = 2.0   # seconds between API calls
SYSTEM_DELAY   = 5.0   # seconds between systems

SYSTEMS = [
    ("system1_gpt_only",     run_gpt_only_query,     {}),
    ("system2_basic_rag",    run_basic_rag_query,     {}),
    ("system3_single_agent", run_single_agent_query,  {}),
    ("system4_agentic_rag",  run_agentic_rag_query,   {}),
]


# ── RAGAS scoring ──────────────────────────────────────────────────
def compute_ragas_scores(question, answer, contexts, ground_truth):
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini"),
            api_version="2024-02-01"
        )
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT",
                                        "text-embedding-3-small"),
            api_version="2024-02-01"
        )

        # Baseline 1 has no contexts — faithfulness = 0
        ctx = contexts if contexts else ["No retrieval context available."]

        dataset = Dataset.from_dict({
            "question":    [question],
            "answer":      [answer],
            "contexts":    [ctx],
            "ground_truth": [ground_truth or question],
        })

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False
        )
        df = result.to_pandas()

        faith = 0.0 if not contexts else float(df["faithfulness"].iloc[0])
        relev = float(df["answer_relevancy"].iloc[0])

        return {
            "faithfulness":     round(faith, 4),
            "answer_relevance": round(relev, 4),
            "note": "faithfulness=0: no retrieval" if not contexts else ""
        }

    except Exception as e:
        print(f"    [RAGAS] Error: {e} — using fallback")
        return {
            "faithfulness":     _faith_fallback(answer, contexts),
            "answer_relevance": _relev_fallback(question, answer),
            "note": f"fallback: {str(e)[:60]}"
        }


def _faith_fallback(answer, contexts):
    if not contexts or not answer:
        return 0.0
    ctx = " ".join(contexts).lower()
    stop = {"the","a","an","is","are","was","in","of","to","for","and","or"}
    words = [w for w in answer.lower().split() if len(w)>4 and w not in stop]
    if not words:
        return 0.5
    return round(min(1.0, sum(1 for w in words if w in ctx)/len(words)+0.1), 4)


def _relev_fallback(question, answer):
    if not answer or not question:
        return 0.0
    q = set(question.lower().split())
    a = set(answer.lower().split())
    return round(min(1.0, len(q & a)/max(len(q),1)+0.3), 4)


# ── Main evaluation loop ───────────────────────────────────────────
def run_evaluation():
    print("=" * 60)
    print("RQ1 RAGAS Evaluation — 4 Systems × 50 Questions")
    print("Metrics: RAGAS Faithfulness + Answer Relevance")
    print("=" * 60)

    model = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini")
    if "mini" in model.lower():
        print(f"\n⚠️  AZURE_CHAT_DEPLOYMENT = {model}")
        print("   For formal evaluation switch to gpt-4o in .env")
        print("   Press Enter to continue or Ctrl+C to stop.")
        input()

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    questions = benchmark["questions"]
    print(f"\nLoaded {len(questions)} benchmark questions\n")

    all_results = {}
    csv_rows    = []

    for sys_name, sys_fn, sys_kwargs in SYSTEMS:
        print(f"\n{'─'*60}")
        print(f"Running: {sys_name}")
        print(f"{'─'*60}")
        system_results = []

        for i, q in enumerate(questions):
            qid   = q["id"]
            query = q["question"]
            gt    = q.get("ground_truth_answer", "")
            qtype = q.get("question_type", "")
            diff  = q.get("difficulty", "")

            print(f"  [{i+1:02d}/50] {qid}: {query[:55]}...")

            try:
                result   = sys_fn(query=query, **sys_kwargs)
                answer   = result.get("answer", "")
                contexts = result.get("contexts", [])
                latency  = result.get("latency_seconds", 0)
                scores   = compute_ragas_scores(query, answer, contexts, gt)

                row = {
                    "system":           sys_name,
                    "id":               qid,
                    "question_type":    qtype,
                    "difficulty":       diff,
                    "faithfulness":     scores["faithfulness"],
                    "answer_relevance": scores["answer_relevance"],
                    "latency":          latency,
                    "note":             scores.get("note", ""),
                }
                system_results.append(row)
                csv_rows.append(row)
                print(f"    F={scores['faithfulness']:.3f}  "
                      f"R={scores['answer_relevance']:.3f}  ({latency}s)")
                time.sleep(CALL_DELAY)

            except Exception as e:
                print(f"    ERROR: {e}")
                row = {"system": sys_name, "id": qid, "question_type": qtype,
                       "difficulty": diff, "faithfulness": 0.0,
                       "answer_relevance": 0.0, "latency": 0,
                       "note": f"error: {str(e)[:60]}"}
                system_results.append(row)
                csv_rows.append(row)

        all_results[sys_name] = system_results
        print(f"\n✅ {sys_name} — {len(system_results)} questions scored")
        time.sleep(SYSTEM_DELAY)

    # Summary statistics
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY — Table 5.1")
    print(f"{'='*60}")
    print(f"{'System':<30} {'Faithfulness':<18} {'Ans Relevance'}")
    print("─" * 60)

    summary = {}
    for sys_name, results in all_results.items():
        fs = [r["faithfulness"]     for r in results]
        rs = [r["answer_relevance"] for r in results]
        mf = statistics.mean(fs);  sf = statistics.stdev(fs) if len(fs)>1 else 0
        mr = statistics.mean(rs);  sr = statistics.stdev(rs) if len(rs)>1 else 0
        summary[sys_name] = {
            "faithfulness_mean": round(mf,4), "faithfulness_sd": round(sf,4),
            "answer_relevance_mean": round(mr,4), "answer_relevance_sd": round(sr,4),
            "n": len(fs)
        }
        print(f"{sys_name:<30} {mf:.4f} (±{sf:.4f})   {mr:.4f} (±{sr:.4f})")

    # Save all outputs
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fields = ["system","id","question_type","difficulty",
                  "faithfulness","answer_relevance","latency","note"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in csv_rows:
            w.writerow({k: row.get(k,"") for k in fields})

    print(f"\n✅ Saved: {OUTPUT_JSON}")
    print(f"✅ Saved: {SUMMARY_JSON}")
    print(f"✅ Saved: {OUTPUT_CSV}")
    print("\nNext step: py statistical_analysis.py")


if __name__ == "__main__":
    run_evaluation()
