"""
p5_scorer_rq2.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

Purpose
-------
Computes Precision@5 for kb-256 / kb-512 / kb-1024 using the verified
benchmark (benchmark_50_questions_verified.json, produced by
resolve_ground_truth_pages.py) and the retrieval results produced by
retrieval_eval_rq2.py.

Relevance rule
--------------
A retrieved chunk is "relevant" to a question if:
  1. chunk.source_document matches one of the question's ground-truth
     document(s), AND
  2. chunk.source_page is within PAGE_TOLERANCE pages of any page listed
     in the question's ground-truth source_page

PAGE_TOLERANCE defaults to 1. This exists because chunking can place the
end of one section's content at the top of the next page, or a chunk may
start a few lines before a heading actually appears — a strict "must be
the EXACT page" rule would unfairly penalise a chunk that's genuinely
useful. Report this tolerance value explicitly in Chapter 4.3's methods
description so examiners know the scoring rule, not just the result.

Output
------
p5_report_rq2.json with:
  - overall mean P@5 + standard deviation per config
  - breakdown by question_type and difficulty
  - worst-10 questions per config for manual error analysis
  - a list of any questions excluded from scoring (unresolved ground truth)
"""

import json
import statistics
from collections import defaultdict

PAGE_TOLERANCE = 1

BENCHMARK_FILE = "benchmark_50_questions_verified.json"
RETRIEVAL_RESULTS_FILE = "retrieval_results_rq2.json"
OUTPUT_REPORT_FILE = "p5_report_rq2.json"


def normalize_doc(name: str) -> str:
    return name.strip().lower()


def get_ground_truth_pairs(question: dict) -> list:
    """
    Returns a list of (document_name, [allowed_pages]) pairs, handling both
    single-document and multi-document (multi_hop) questions uniformly.
    """
    gt_documents = [d.strip() for d in question["source_document"].split(";")]
    source_page = question.get("source_page")

    if len(gt_documents) == 1:
        pages = source_page if isinstance(source_page, list) else ([source_page] if source_page else [])
        return [(gt_documents[0], pages)]

    # multi-hop: source_page is a list-of-lists, one per document, in order
    pairs = []
    for i, doc in enumerate(gt_documents):
        pages = source_page[i] if source_page and i < len(source_page) else []
        pages = pages if isinstance(pages, list) else ([pages] if pages else [])
        pairs.append((doc, pages))
    return pairs


def is_relevant(chunk: dict, ground_truth_pairs: list) -> bool:
    chunk_doc = normalize_doc(chunk.get("source_document", ""))
    chunk_page = chunk.get("source_page")
    if chunk_page is None:
        return False

    for gt_doc, gt_pages in ground_truth_pairs:
        if normalize_doc(gt_doc) != chunk_doc:
            continue
        for gt_page in gt_pages:
            if gt_page is None:
                continue
            if abs(int(chunk_page) - int(gt_page)) <= PAGE_TOLERANCE:
                return True
    return False


def score_question(retrieved_chunks: list, question: dict) -> float:
    if not retrieved_chunks:
        return 0.0
    ground_truth_pairs = get_ground_truth_pairs(question)
    # if every page list is empty, ground truth is unresolved -> caller should exclude
    if all(len(pages) == 0 for _, pages in ground_truth_pairs):
        return None
    relevant_count = sum(1 for c in retrieved_chunks[:5] if is_relevant(c, ground_truth_pairs))
    return relevant_count / min(5, len(retrieved_chunks))


def run():
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    with open(RETRIEVAL_RESULTS_FILE, "r", encoding="utf-8") as f:
        retrieval_results = json.load(f)

    questions_by_id = {q["id"]: q for q in benchmark["questions"]}
    report = {}
    excluded_questions = []

    for kb_name, kb_results in retrieval_results.items():
        per_question_scores = {}

        for qid, chunks in kb_results.items():
            question = questions_by_id[qid]
            score = score_question(chunks, question)
            if score is None:
                if qid not in excluded_questions:
                    excluded_questions.append(qid)
                continue
            per_question_scores[qid] = score

        scores = list(per_question_scores.values())
        overall_mean = statistics.mean(scores) if scores else 0.0
        overall_sd = statistics.stdev(scores) if len(scores) > 1 else 0.0

        by_type = defaultdict(list)
        by_difficulty = defaultdict(list)
        for qid, score in per_question_scores.items():
            q = questions_by_id[qid]
            by_type[q["question_type"]].append(score)
            by_difficulty[q["difficulty"]].append(score)

        type_breakdown = {
            t: {
                "mean": round(statistics.mean(s), 3),
                "sd": round(statistics.stdev(s), 3) if len(s) > 1 else 0.0,
                "n": len(s),
            }
            for t, s in by_type.items()
        }
        difficulty_breakdown = {
            d: {
                "mean": round(statistics.mean(s), 3),
                "sd": round(statistics.stdev(s), 3) if len(s) > 1 else 0.0,
                "n": len(s),
            }
            for d, s in by_difficulty.items()
        }

        worst_10 = sorted(per_question_scores.items(), key=lambda x: x[1])[:10]
        worst_10_detail = [
            {
                "id": qid,
                "score": score,
                "question": questions_by_id[qid]["question"],
                "expected_source": f"{questions_by_id[qid]['source_document']} (page {questions_by_id[qid].get('source_page')})",
            }
            for qid, score in worst_10
        ]

        report[kb_name] = {
            "overall_mean_p5": round(overall_mean, 3),
            "overall_sd": round(overall_sd, 3),
            "n_questions_scored": len(scores),
            "page_tolerance_used": PAGE_TOLERANCE,
            "breakdown_by_question_type": type_breakdown,
            "breakdown_by_difficulty": difficulty_breakdown,
            "worst_10_for_error_analysis": worst_10_detail,
        }

    report["_excluded_questions_unresolved_ground_truth"] = excluded_questions

    with open(OUTPUT_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== P@5 SUMMARY (RQ2) ===")
    for kb_name in retrieval_results.keys():
        res = report[kb_name]
        print(f"{kb_name}: mean={res['overall_mean_p5']}  sd={res['overall_sd']}  "
              f"n={res['n_questions_scored']}/50")

    if excluded_questions:
        print(f"\nWARNING: {len(excluded_questions)} question(s) excluded from scoring "
              f"due to unresolved ground truth: {excluded_questions}")
        print("Fix these in benchmark_50_questions_verified.json and re-run.")

    scored_configs = {k: v for k, v in report.items() if not k.startswith("_")}
    if scored_configs:
        best_kb = max(scored_configs.items(), key=lambda x: x[1]["overall_mean_p5"])[0]
        print(f"\nHighest average P@5: {best_kb} -> candidate for frozen retrieval config")

    print(f"Full report saved to {OUTPUT_REPORT_FILE}")


if __name__ == "__main__":
    run()
