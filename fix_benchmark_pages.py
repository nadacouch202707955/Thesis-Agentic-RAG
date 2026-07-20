"""
fix_benchmark_pages.py
Updates source_page fields in benchmark_50_questions_verified.json
based on manually verified page numbers from PDF lookup.

Run: py fix_benchmark_pages.py
"""

import json

# Manually verified page numbers
# Format: "Q_ID": [page1] or [page1, page2] for multi-page answers
PAGE_UPDATES = {
    "Q033": [5],        # Academic_Calendar.pdf — "August 2026"
    "Q034": [2],        # Academic_Calendar.pdf — "Classes normally run Sunday"
    "Q035": [3],        # Academic_Calendar.pdf — "October 2025"
    "Q036": [7],        # Student-Handbook.pdf — "Useful Contacts"
    "Q037": [13],       # Student-Handbook.pdf — "Health and Wellness Centre"
    "Q038": [15],       # Student-Handbook.pdf — "Library Services" (first)
    "Q039": [2],        # Student-Handbook.pdf — "Library Services" (second)
    "Q040": [6],        # Student-Handbook.pdf — "Our core values are Excellence"
    "Q041": [8],        # Student-Handbook.pdf — "Academic Advising"
    "Q042": [10],       # Student-Handbook.pdf — "Academic Facilities and Resources"
    "Q043": [21],       # Student-Handbook.pdf — "Academic Misconduct"
    "Q044": [22],       # Student-Handbook.pdf — "Appeals to Faculty Appeal Committee"
    "Q045": [2],        # Student-Handbook.pdf — "Library Services" (third)
    "Q046": [2, 17, 32],# Student-Handbook.pdf — "Scholarships and Fees" (multi-page)
    "Q047": [12],       # Student-Handbook.pdf — "Polytechnic security staff"
    "Q048": [2, 8],     # Student-Handbook.pdf — "Definitions of Employability Skills"
    "Q049": [16],       # Student-Handbook.pdf — "Learning Support"
    "Q050": [8],        # Student-Handbook.pdf — "Counseling/Academic Advising/Mentoring"
                        # NOTE: exact section not found — using Academic Advising page (p.8)
                        # as closest match. Q050 may score 0 in P@5 if retrieval misses p.8.
}

INPUT_FILE  = "benchmark_50_questions_verified.json"
OUTPUT_FILE = "benchmark_50_questions_verified.json"  # overwrites in place

def fix_pages():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    updated = 0
    skipped = []

    for question in benchmark["questions"]:
        qid = question.get("id", "")
        if qid in PAGE_UPDATES:
            pages = PAGE_UPDATES[qid]
            # Set source_page — use first page as primary, store all as list
            question["source_page"] = pages[0]
            question["source_pages_all"] = pages  # extra field for multi-page answers
            updated += 1
            print(f"  ✅ {qid} → source_page={pages[0]}  (all: {pages})")
        else:
            # Q001–Q032 should already be resolved — leave them alone
            if question.get("source_page") is None or question.get("source_page") == 0:
                skipped.append(qid)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] Updated {updated} questions in {OUTPUT_FILE}")
    if skipped:
        print(f"[Warning] {len(skipped)} questions still have no page: {skipped}")
    print("\nNext step: py p5_scorer_rq2.py")

if __name__ == "__main__":
    fix_pages()
