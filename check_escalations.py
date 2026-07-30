"""
check_escalations.py
Quick diagnostic: how many of Agentic RAG's 50 responses were HITL
escalations, and how much do they drag down the Faithfulness /
Answer Relevance averages?

Run this from your project folder (Thesis-Agentic-Rag), inside venv311:
    python check_escalations.py

Requires:
    - ragas_results_all_systems.json  (from ragas_eval_rq1.py)
    - escalations.json                (written by orchestrator_agent.py)
"""

import json
import statistics

RAGAS_FILE       = "ragas_results_all_systems.json"
ESCALATIONS_FILE = "escalations.json"

# Standard phrase your Notification Agent uses for escalated responses
ESCALATION_MARKER = "forwarded to an academic advisor"


def main():
    with open(RAGAS_FILE, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    agentic_rows = all_results.get("system4_agentic_rag", [])
    if not agentic_rows:
        print("No system4_agentic_rag results found in", RAGAS_FILE)
        return

    print("=" * 60)
    print(f"system4_agentic_rag — {len(agentic_rows)} total questions")
    print("=" * 60)

    # ── Try escalations.json first (most reliable — logged by the pipeline itself)
    escalated_ids = set()
    try:
        with open(ESCALATIONS_FILE, "r", encoding="utf-8") as f:
            escalations = json.load(f)
        for rec in escalations:
            q = rec.get("query", "")
            escalated_ids.add(q.strip())
        print(f"\n[escalations.json] {len(escalations)} escalation records found "
              f"(across ALL runs/systems, not just this eval)")
    except FileNotFoundError:
        print(f"\n[escalations.json] not found — will rely on notes/answer text instead")

    # ── Cross-check using the "note" field or low scores as a proxy
    # (ragas_results doesn't store the raw answer text, so we approximate
    #  escalation by looking for suspiciously low faithfulness combined
    #  with the standard low-confidence pattern via the note field)
    escalated_rows = []
    normal_rows    = []

    for row in agentic_rows:
        note = (row.get("note") or "").lower()
        # Fallback heuristic: RAGAS-scored escalation responses tend to
        # score very low on both metrics because they're generic text
        # not grounded in retrieved context. We flag rows with
        # faithfulness == 0 as *candidates* for manual/escalation review.
        if row["faithfulness"] == 0.0:
            escalated_rows.append(row)
        else:
            normal_rows.append(row)

    print(f"\nRows with Faithfulness = 0.0 (candidate escalations/failures): "
          f"{len(escalated_rows)} / {len(agentic_rows)}")
    for r in escalated_rows:
        print(f"  {r['id']}: F={r['faithfulness']:.3f}  R={r['answer_relevance']:.3f}  "
              f"note={r.get('note','')[:50]}")

    # ── Compare: all 50 vs excluding the zero-faithfulness rows
    all_f = [r["faithfulness"]     for r in agentic_rows]
    all_r = [r["answer_relevance"] for r in agentic_rows]
    norm_f = [r["faithfulness"]     for r in normal_rows]
    norm_r = [r["answer_relevance"] for r in normal_rows]

    print("\n" + "=" * 60)
    print("Faithfulness / Relevance — ALL 50 vs EXCLUDING zero-faithfulness rows")
    print("=" * 60)
    print(f"All 50 questions:        F_mean={statistics.mean(all_f):.4f}  "
          f"R_mean={statistics.mean(all_r):.4f}")
    if norm_f:
        print(f"Excluding {len(escalated_rows)} zero-F rows: "
              f"F_mean={statistics.mean(norm_f):.4f}  "
              f"R_mean={statistics.mean(norm_r):.4f}  (n={len(norm_f)})")
    else:
        print("No rows remain after exclusion — check data.")

    print("\nNote: Faithfulness=0.0 can also occur for genuinely poor answers,")
    print("not only HITL escalations. Cross-check the listed question IDs above")
    print("against your smoke-test / escalations.json output to confirm which")
    print("ones are true escalations before citing this in Chapter 5.")


if __name__ == "__main__":
    main()
