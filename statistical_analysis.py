"""
statistical_analysis.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

RQ1 Statistical Analysis — Paired t-test + Effect Size
=======================================================

PURPOSE
-------
This script answers the statistical hypothesis in RQ1:

    H1: There is a significant difference in RAGAS Faithfulness
        between Agentic RAG and Conventional RAG (Baseline 2).

    H2: There is a significant difference in RAGAS Answer Relevance
        between Agentic RAG and Conventional RAG (Baseline 2).

Tests performed:
    1. Paired t-test (scipy.stats.ttest_rel)
       - Paired because the same 50 questions are used for both systems
       - Two-tailed, significance level α = 0.05

    2. Cohen's d effect size
       - Small: d < 0.2
       - Medium: 0.2 ≤ d < 0.8
       - Large: d ≥ 0.8

    3. Descriptive statistics for all 4 systems

Input:  ragas_results_all_systems.json (from ragas_eval_rq1.py)
Output: statistical_results.json (for Chapter 5 §5.5)
        statistical_results.csv  (for thesis appendix)

Run: py statistical_analysis.py
"""

import json
import csv
import statistics
import math
from scipy.stats import ttest_rel, shapiro

INPUT_FILE  = "ragas_results_all_systems.json"
OUTPUT_JSON = "statistical_results.json"
OUTPUT_CSV  = "statistical_results.csv"

ALPHA = 0.05   # significance level


def cohen_d(group1: list, group2: list) -> float:
    """
    Computes Cohen's d effect size for two paired groups.
    Uses pooled standard deviation.
    """
    n     = len(group1)
    diff  = [a - b for a, b in zip(group1, group2)]
    mean_diff = statistics.mean(diff)
    sd_diff   = statistics.stdev(diff) if n > 1 else 0.0
    return round(mean_diff / sd_diff, 4) if sd_diff != 0 else 0.0


def interpret_effect(d: float) -> str:
    d = abs(d)
    if d < 0.2:   return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else:          return "large"


def interpret_p(p: float) -> str:
    if p < 0.001: return "p < 0.001 — highly significant"
    elif p < 0.01: return "p < 0.01 — very significant"
    elif p < 0.05: return "p < 0.05 — significant"
    else:          return f"p = {p:.4f} — not significant (fail to reject H0)"


def run():
    print("=" * 60)
    print("RQ1 Statistical Analysis")
    print("Paired t-test + Cohen's d Effect Size")
    print("=" * 60)

    # Load RAGAS results
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    # Extract scores per system (ordered by question ID)
    systems = {}
    for sys_name, results in all_results.items():
        sorted_results = sorted(results, key=lambda x: x["id"])
        systems[sys_name] = {
            "faithfulness":     [r["faithfulness"]     for r in sorted_results],
            "answer_relevance": [r["answer_relevance"] for r in sorted_results],
        }

    # ── Descriptive statistics ─────────────────────────────────────
    print("\n--- Descriptive Statistics ---")
    print(f"{'System':<30} {'Metric':<20} {'Mean':<8} {'SD':<8} {'Min':<8} {'Max'}")
    print("─" * 75)

    descriptive = {}
    for sys_name, scores in systems.items():
        for metric in ["faithfulness", "answer_relevance"]:
            vals = scores[metric]
            mean = statistics.mean(vals)
            sd   = statistics.stdev(vals) if len(vals) > 1 else 0.0
            mn   = min(vals)
            mx   = max(vals)
            descriptive[f"{sys_name}_{metric}"] = {
                "mean": round(mean, 4), "sd": round(sd, 4),
                "min":  round(mn, 4),  "max": round(mx, 4), "n": len(vals)
            }
            print(f"{sys_name:<30} {metric:<20} {mean:.4f}  {sd:.4f}  "
                  f"{mn:.4f}  {mx:.4f}")

    # ── Normality check (Shapiro-Wilk) ────────────────────────────
    print("\n--- Normality Check (Shapiro-Wilk) ---")
    normality = {}
    sys2_f = systems.get("system2_basic_rag", {}).get("faithfulness", [])
    sys4_f = systems.get("system4_agentic_rag", {}).get("faithfulness", [])
    sys2_r = systems.get("system2_basic_rag", {}).get("answer_relevance", [])
    sys4_r = systems.get("system4_agentic_rag", {}).get("answer_relevance", [])

    for label, data in [
        ("System2 Faithfulness", sys2_f),
        ("System4 Faithfulness", sys4_f),
        ("System2 Ans Relevance", sys2_r),
        ("System4 Ans Relevance", sys4_r),
    ]:
        if len(data) >= 3:
            stat, p = shapiro(data)
            normal = bool(p > 0.05)
            normality[label] = {"W": round(stat,4), "p": round(p,4),
                                 "normal": normal}
            print(f"  {label:<25} W={stat:.4f}  p={p:.4f}  "
                  f"{'Normal' if normal else 'Non-normal'}")

    # ── Paired t-test: System 2 vs System 4 ───────────────────────
    print("\n--- Paired t-test: Conventional RAG vs Agentic RAG ---")
    hypothesis_tests = {}

    for metric, s2, s4, h_label in [
        ("faithfulness",     sys2_f, sys4_f, "H1"),
        ("answer_relevance", sys2_r, sys4_r, "H2"),
    ]:
        if not s2 or not s4:
            print(f"  {h_label} ({metric}): insufficient data")
            continue

        t_stat, p_val = ttest_rel(s2, s4)
        d             = cohen_d(s4, s2)   # Agentic minus Basic
        effect_label  = interpret_effect(d)
        p_interp      = interpret_p(p_val)
        reject_h0     = bool(p_val < ALPHA)

        hypothesis_tests[h_label] = {
            "metric":        metric,
            "t_statistic":   round(t_stat, 4),
            "p_value":       round(p_val,  6),
            "cohen_d":       d,
            "effect_size":   effect_label,
            "reject_h0":     reject_h0,
            "alpha":         ALPHA,
            "interpretation": p_interp,
            "mean_system2":  round(statistics.mean(s2), 4),
            "mean_system4":  round(statistics.mean(s4), 4),
            "difference":    round(statistics.mean(s4)-statistics.mean(s2), 4),
        }

        print(f"\n  {h_label} — {metric.upper()}")
        print(f"    System 2 (Basic RAG) mean:   {statistics.mean(s2):.4f}")
        print(f"    System 4 (Agentic RAG) mean: {statistics.mean(s4):.4f}")
        print(f"    Difference:                  {statistics.mean(s4)-statistics.mean(s2):+.4f}")
        print(f"    t-statistic: {t_stat:.4f}")
        print(f"    p-value:     {p_val:.6f}  →  {p_interp}")
        print(f"    Cohen's d:   {d:.4f}  ({effect_label} effect)")
        print(f"    Decision:    {'✅ Reject H0 — significant difference' if reject_h0 else '❌ Fail to reject H0 — no significant difference'}")

    # ── Additional comparisons ─────────────────────────────────────
    print("\n--- Additional Pairwise Comparisons (Baseline 1 vs Agentic RAG) ---")
    sys1_f = systems.get("system1_gpt_only", {}).get("faithfulness", [])
    sys1_r = systems.get("system1_gpt_only", {}).get("answer_relevance", [])

    additional = {}
    for metric, s1, s4 in [
        ("faithfulness",     sys1_f, sys4_f),
        ("answer_relevance", sys1_r, sys4_r),
    ]:
        if s1 and s4:
            t_stat, p_val = ttest_rel(s1, s4)
            d = cohen_d(s4, s1)
            additional[f"sys1_vs_sys4_{metric}"] = {
                "t": round(t_stat,4), "p": round(p_val,6),
                "d": d, "effect": interpret_effect(d)
            }
            print(f"  GPT-only vs Agentic RAG ({metric}): "
                  f"t={t_stat:.4f}  p={p_val:.6f}  d={d:.4f} ({interpret_effect(d)})")

    # ── Save results ───────────────────────────────────────────────
    output = {
        "descriptive_statistics": descriptive,
        "normality_tests":        normality,
        "hypothesis_tests":       hypothesis_tests,
        "additional_comparisons": additional,
        "analysis_settings": {
            "alpha":       ALPHA,
            "test":        "paired t-test (ttest_rel)",
            "effect_size": "Cohen's d",
            "n_questions": 50,
            "systems_compared": "System 2 (Conventional RAG) vs System 4 (Agentic RAG)"
        }
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # CSV for appendix
    csv_rows = []
    for sys_name, scores in systems.items():
        for i, (f, r) in enumerate(zip(
            scores["faithfulness"], scores["answer_relevance"]
        )):
            csv_rows.append({
                "system": sys_name, "question_index": i+1,
                "faithfulness": f, "answer_relevance": r
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "system","question_index","faithfulness","answer_relevance"])
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\n✅ Saved: {OUTPUT_JSON}")
    print(f"✅ Saved: {OUTPUT_CSV}")
    print("\nUse statistical_results.json to write Chapter 5 §5.5")
    print("Key values needed for thesis:")
    for h, res in hypothesis_tests.items():
        print(f"  {h}: t={res['t_statistic']}, p={res['p_value']}, "
              f"d={res['cohen_d']} ({res['effect_size']}), "
              f"reject_H0={res['reject_h0']}")


if __name__ == "__main__":
    run()
