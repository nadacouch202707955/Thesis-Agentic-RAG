"""
p5_scorer_rq2.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

Computes Precision@5 for kb-256 / kb-512 / kb-1024 using the verified
benchmark (benchmark_50_questions_verified.json) and retrieval results
from retrieval_eval_rq2.py.

Relevance rule
--------------
A retrieved chunk is "relevant" if:
  - chunk.source_document matches the question's ground-truth document, AND
  - If source_page IS known: chunk.source_page is within PAGE_TOLERANCE pages
  - If source_page IS NULL:  document-only match is accepted (Q001-Q032)

PAGE_TOLERANCE = 1 (one page either side of ground truth page).
Document-only matching for null-page questions is noted as a study
limitation in Chapter 5 error analysis.
"""

import json
import statistics
from collections import defaultdict

PAGE_TOLERANCE = 1

BENCHMARK_FILE        = "benchmark_50_questions_verified.json"
RETRIEVAL_RESULTS_FILE = "retrieval_results_rq2.json"
OUTPUT_REPORT_FILE    = "p5_report_rq2.json"
OUTPUT_CSV_FILE       = "p5_results_rq2.csv"


def normalize_doc(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def get_ground_truth_pairs(question: dict) -> list:
    """
    Returns list of (document_name, [allowed_pages_or_empty]) pairs.
    Empty pages list = document-only matching (source_page was null).
    """
    gt_documents = [d.strip() for d in question["source_document"].split(";")]
    source_page  = question.get("source_page")

    if len(gt_documents) == 1:
        if source_page is None:
            pages = []           # document-only match
        elif isinstance(source_page, list):
            pages = source_page
        else:
            pages = [source_page]
        return [(gt_documents[0], pages)]

    # multi-hop questions
    pairs = []
    for i, doc in enumerate(gt_documents):
        if source_page is None:
            pg = []
        elif isinstance(source_page, list):
            p = source_page[i] if i < len(source_page) else None
            pg = p if isinstance(p, list) else ([p] if p is not None else [])
        else:
            pg = []
        pairs.append((doc, pg))
    return pairs


def is_relevant(chunk: dict, ground_truth_pairs: list) -> bool:
    """
    Returns True if the chunk is relevant to any ground-truth (doc, pages) pair.
    When pages is empty → document-only match (source_page was null).
    """
    chunk_doc  = normalize_doc(chunk.get("source_document", ""))
    chunk_page = chunk.get("source_page")

    for gt_doc, gt_pages in ground_truth_pairs:
        if normalize_doc(gt_doc) != chunk_doc:
            continue

        # document-only match (source_page was null in benchmark)
        if len(gt_pages) == 0:
            return True

        # page-aware match
        if chunk_page is None:
            continue
        for gt_page in gt_pages:
            if gt_page is None:
                continue
            if abs(int(chunk_page) - int(gt_page)) <= PAGE_TOLERANCE:
                return True

    return False


def score_question(retrieved_chunks: list, question: dict) -> float:
    """
    Returns P@5 score for one question.
    Returns None only if source_document is missing (truly unresolvable).
    """
    if not retrieved_chunks:
        return 0.0

    # Only exclude if source_document itself is missing
    if not question.get("source_document"):
        return None

    ground_truth_pairs = get_ground_truth_pairs(question)
    relevant_count = sum(
        1 for c in retrieved_chunks[:5]
        if is_relevant(c, ground_truth_pairs)
    )
    return relevant_count / min(5, len(retrieved_chunks))


def run():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    with open(RETRIEVAL_RESULTS_FILE, "r", encoding="utf-8") as f:
        retrieval_results = json.load(f)

    questions_by_id = {q["id"]: q for q in benchmark["questions"]}
    report           = {}
    excluded_questions = []

    # Track per-question scores across all KBs for CSV export
    csv_rows = []

    for kb_name, kb_results in retrieval_results.items():
        per_question_scores = {}

        for qid, chunks in kb_results.items():
            question = questions_by_id.get(qid)
            if question is None:
                continue
            score = score_question(chunks, question)
            if score is None:
                if qid not in excluded_questions:
                    excluded_questions.append(qid)
                continue
            per_question_scores[qid] = score
            csv_rows.append({
                "kb": kb_name,
                "id": qid,
                "score": round(score, 3),
                "question_type": question.get("question_type", ""),
                "difficulty": question.get("difficulty", ""),
                "source_page_known": question.get("source_page") is not None,
            })

        scores       = list(per_question_scores.values())
        overall_mean = statistics.mean(scores)   if scores         else 0.0
        overall_sd   = statistics.stdev(scores)  if len(scores) > 1 else 0.0

        by_type       = defaultdict(list)
        by_difficulty = defaultdict(list)
        for qid, score in per_question_scores.items():
            q = questions_by_id[qid]
            by_type[q["question_type"]].append(score)
            by_difficulty[q["difficulty"]].append(score)

        type_breakdown = {
            t: {
                "mean": round(statistics.mean(s), 3),
                "sd":   round(statistics.stdev(s), 3) if len(s) > 1 else 0.0,
                "n":    len(s),
            }
            for t, s in by_type.items()
        }
        difficulty_breakdown = {
            d: {
                "mean": round(statistics.mean(s), 3),
                "sd":   round(statistics.stdev(s), 3) if len(s) > 1 else 0.0,
                "n":    len(s),
            }
            for d, s in by_difficulty.items()
        }

        worst_10 = sorted(per_question_scores.items(), key=lambda x: x[1])[:10]
        worst_10_detail = [
            {
                "id":              qid,
                "score":           score,
                "question":        questions_by_id[qid]["question"],
                "expected_source": (
                    f"{questions_by_id[qid]['source_document']} "
                    f"(page {questions_by_id[qid].get('source_page', 'unknown')})"
                ),
            }
            for qid, score in worst_10
        ]

        report[kb_name] = {
            "overall_mean_p5":            round(overall_mean, 3),
            "overall_sd":                 round(overall_sd, 3),
            "n_questions_scored":         len(scores),
            "page_tolerance_used":        PAGE_TOLERANCE,
            "note_null_page_questions":   "Q001-Q032 used document-only matching (source_page was null)",
            "breakdown_by_question_type": type_breakdown,
            "breakdown_by_difficulty":    difficulty_breakdown,
            "worst_10_for_error_analysis": worst_10_detail,
        }

    report["_excluded_questions"] = excluded_questions

    # Save JSON report
    with open(OUTPUT_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save CSV for thesis appendix
    import csv
    with open(OUTPUT_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kb","id","score","question_type","difficulty","source_page_known"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Print summary
    print("\n=== P@5 SUMMARY (RQ2) ===")
    for kb_name in retrieval_results.keys():
        if kb_name not in report:
            continue
        res = report[kb_name]
        print(f"{kb_name}: mean={res['overall_mean_p5']}  "
              f"sd={res['overall_sd']}  "
              f"n={res['n_questions_scored']}/50")

    if excluded_questions:
        print(f"\nWARNING: {len(excluded_questions)} question(s) excluded "
              f"(source_document missing): {excluded_questions}")

    scored_configs = {k: v for k, v in report.items() if not k.startswith("_")}
    if scored_configs:
        best_kb = max(scored_configs.items(),
                      key=lambda x: x[1]["overall_mean_p5"])[0]
        best_score = report[best_kb]["overall_mean_p5"]
        print(f"\n★ WINNER: {best_kb}  (mean P@5 = {best_score})")
        print(f"  → Use this as your frozen retrieval config for Agentic RAG")
        print(f"\nFull report: {OUTPUT_REPORT_FILE}")
        print(f"CSV for thesis appendix: {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
    run()
