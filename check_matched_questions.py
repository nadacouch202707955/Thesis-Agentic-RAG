"""
check_matched_questions.py
Read-only analysis — does NOT modify any existing files.

Checks how system2_basic_rag and system3_single_agent scored on the
exact same 12 question IDs that system4_agentic_rag escalated (scored
Faithfulness = 0.0), to test whether the "Agentic RAG outperforms
baselines on delivered answers" claim holds on a fair, matched subset.

Run this from your project folder (Thesis-Agentic-Rag):
    python check_matched_questions.py

Requires: ragas_results_all_systems.json (already saved from your last run)
"""

import json
import statistics

RAGAS_FILE = "ragas_results_all_systems.json"


def main():
    with open(RAGAS_FILE, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    agentic_rows = all_results.get("system4_agentic_rag", [])
    basic_rows   = all_results.get("system2_basic_rag", [])
    single_rows  = all_results.get("system3_single_agent", [])

    # Identify the 12 escalated question IDs (Faithfulness == 0.0 in Agentic RAG)
    escalated_ids = {r["id"] for r in agentic_rows if r["faithfulness"] == 0.0}
    answered_ids  = {r["id"] for r in agentic_rows if r["faithfulness"] != 0.0}

    print("=" * 65)
    print(f"Agentic RAG escalated {len(escalated_ids)} questions: {sorted(escalated_ids)}")
    print("=" * 65)

    def scores_for(rows, ids):
        matched = [r for r in rows if r["id"] in ids]
        f_vals = [r["faithfulness"] for r in matched]
        r_vals = [r["answer_relevance"] for r in matched]
        return matched, f_vals, r_vals

    # ── Part 1: How did baselines score on the SAME 12 "hard" questions? ──
    print("\n--- Baseline performance on the SAME 12 questions Agentic RAG escalated ---")
    for name, rows in [("system2_basic_rag", basic_rows), ("system3_single_agent", single_rows)]:
        matched, f_vals, r_vals = scores_for(rows, escalated_ids)
        if f_vals:
            print(f"{name}: n={len(matched)}  "
                  f"F_mean={statistics.mean(f_vals):.4f}  R_mean={statistics.mean(r_vals):.4f}")
        else:
            print(f"{name}: no matching rows found (check ID format)")

    # ── Part 2: Fair, matched comparison — same 38 "answered" questions for everyone ──
    print("\n--- Fair comparison: ALL systems on the SAME 38 questions Agentic RAG answered ---")
    for name, rows in [
        ("system2_basic_rag", basic_rows),
        ("system3_single_agent", single_rows),
        ("system4_agentic_rag", agentic_rows),
    ]:
        matched, f_vals, r_vals = scores_for(rows, answered_ids)
        if f_vals:
            print(f"{name}: n={len(matched)}  "
                  f"F_mean={statistics.mean(f_vals):.4f}  R_mean={statistics.mean(r_vals):.4f}")
        else:
            print(f"{name}: no matching rows found (check ID format)")

    print("\nInterpretation guide:")
    print("- Part 1 tells you whether the 12 escalated questions were genuinely")
    print("  hard for everyone (baselines also score low) or only hard for Agentic RAG.")
    print("- Part 2 is the fair, apples-to-apples comparison: same 38 questions,")
    print("  same metric, all 4 systems. This is the number to cite if you want to")
    print("  claim Agentic RAG 'outperforms' baselines on delivered answers.")


if __name__ == "__main__":
    main()
