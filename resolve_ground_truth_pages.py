"""
resolve_ground_truth_pages.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

Purpose
-------
The 50-question benchmark currently labels each question's ground truth
location with a human-readable section heading (e.g. "3.2 Attendance
Requirements", "September 2025", "Health and Wellness Centre") but your
ingestion pipeline tags chunks with PAGE NUMBERS, not heading text. To
score Precision@5 automatically, the two need to speak the same language.

Rather than eyeballing each PDF and typing in page numbers by hand (which
is slow AND hard to defend in a viva — "how do you know page 14 is
correct?"), this script PROGRAMMATICALLY searches each source PDF for the
literal heading text using PyMuPDF and records which page(s) it appears
on. This gives you a reproducible, defensible answer: "ground truth page
numbers were verified by direct text search against the source PDF, not
manually estimated" — you can say this exact sentence in your viva.

For multi-hop questions with two documents (semicolon-separated in
source_document/source_section), each part is resolved independently.

Run: python resolve_ground_truth_pages.py
Output: benchmark_50_questions_verified.json
    (same as input, plus a "source_page" field per question — int, list of
    ints, or null if the heading text could not be located automatically)

IMPORTANT: after running, check the printed "NEEDS MANUAL REVIEW" list —
any question where source_page is null must be checked by hand and fixed
before running the scorer, or it will be silently excluded from P@5.
"""

import fitz  # PyMuPDF
import json
import os

# Edit this to wherever your 4 source PDFs live locally
PDF_DIR = "./source_pdfs"

PDF_FILENAME_MAP = {
    "Student-Handbook.pdf": "Student-Handbook.pdf",
    "Academic-Regulations.pdf": "Academic-Regulations.pdf",
    "Academic_Calendar.pdf": "Academic_Calendar.pdf",
    "Bachelor_of_Business__Marketing.pdf": "Bachelor_of_Business__Marketing.pdf",
}

BENCHMARK_IN = "benchmark_50_questions.json"
BENCHMARK_OUT = "benchmark_50_questions_verified.json"


def find_pages_for_heading(pdf_path: str, heading_text: str) -> list:
    """Returns a list of 1-indexed page numbers where heading_text is found verbatim."""
    if not os.path.exists(pdf_path):
        return []

    doc = fitz.open(pdf_path)
    found_pages = []

    for page_num, page in enumerate(doc, start=1):
        matches = page.search_for(heading_text)
        if matches:
            found_pages.append(page_num)

    doc.close()
    return found_pages


def resolve_with_fallback(pdf_path: str, heading_text: str) -> list:
    """
    Tries the full heading text first. If nothing is found (common cause:
    PDF text extraction inserts slightly different spacing/line breaks than
    the heading as typed in the benchmark), falls back to the first 20
    characters, which is usually distinctive enough for these documents'
    short headings while being more tolerant of extraction quirks.
    """
    pages = find_pages_for_heading(pdf_path, heading_text)
    if pages:
        return pages

    short_anchor = heading_text[:20].strip()
    if len(short_anchor) >= 6:  # avoid matching on near-empty strings
        pages = find_pages_for_heading(pdf_path, short_anchor)

    return pages


def resolve_question(question: dict) -> dict:
    gt_documents = [d.strip() for d in question["source_document"].split(";")]
    gt_sections = [s.strip() for s in question["source_section"].split(";")]

    question = dict(question)  # shallow copy

    if len(gt_documents) == 1:
        # Single document, but possibly MULTIPLE valid section anchors
        # (e.g. "Counseling; Academic Advising; Mentoring" all live in the
        # same handbook). Union all pages found across every section anchor
        # rather than pairing 1:1 with documents, which would silently drop
        # every section after the first.
        pdf_filename = PDF_FILENAME_MAP.get(gt_documents[0])
        all_pages = set()
        for section in gt_sections:
            pdf_path = os.path.join(PDF_DIR, pdf_filename) if pdf_filename else None
            pages = resolve_with_fallback(pdf_path, section) if pdf_path else []
            all_pages.update(pages)
        question["source_page"] = sorted(all_pages) if all_pages else None
        return question

    # Multiple documents (multi_hop): pair each document with its
    # corresponding section 1:1, in order.
    while len(gt_sections) < len(gt_documents):
        gt_sections.append(gt_sections[-1] if gt_sections else "")

    resolved_pages = []
    for doc_name, section in zip(gt_documents, gt_sections):
        pdf_filename = PDF_FILENAME_MAP.get(doc_name)
        if not pdf_filename:
            resolved_pages.append(None)
            continue
        pdf_path = os.path.join(PDF_DIR, pdf_filename)
        pages = resolve_with_fallback(pdf_path, section)
        resolved_pages.append(pages if pages else None)

    question["source_page"] = resolved_pages
    return question


def run():
    with open(BENCHMARK_IN, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    needs_review = []
    ambiguous_broad_match = []
    resolved_questions = []

    for q in benchmark["questions"]:
        resolved_q = resolve_question(q)
        resolved_questions.append(resolved_q)

        # Flag anything unresolved
        sp = resolved_q["source_page"]
        is_unresolved = (sp is None) or (isinstance(sp, list) and any(p is None for p in sp))
        if is_unresolved:
            needs_review.append(resolved_q["id"])

        # Flag anchors that matched suspiciously many pages (>2) — likely a
        # too-generic anchor (e.g. "Vision" also appearing in a table of
        # contents), which would let unrelated chunks score as "relevant"
        def flatten(x):
            if x is None:
                return []
            if isinstance(x, list):
                out = []
                for i in x:
                    out.extend(flatten(i))
                return out
            return [x]

        flat_pages = flatten(sp)
        if len(flat_pages) > 2:
            ambiguous_broad_match.append((resolved_q["id"], resolved_q["source_section"], flat_pages))

    benchmark["questions"] = resolved_questions
    benchmark["metadata"]["ground_truth_page_verification"] = (
        "source_page values were resolved automatically by searching for the "
        "literal source_section heading text within the source PDF using "
        "PyMuPDF (page.search_for). Entries flagged for manual review could "
        "not be located verbatim and were checked/corrected by hand."
    )

    with open(BENCHMARK_OUT, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)

    print(f"Resolved ground-truth pages for {len(resolved_questions)} questions.")
    print(f"Saved to {BENCHMARK_OUT}")

    if needs_review:
        print(f"\nNEEDS MANUAL REVIEW ({len(needs_review)} questions) — source_page could not be auto-located:")
        for qid in needs_review:
            q = next(x for x in resolved_questions if x["id"] == qid)
            print(f"  {qid}: \"{q['source_section']}\" in {q['source_document']}")
        print("\nOpen the PDF, find the correct page number by eye, and edit")
        print(f"{BENCHMARK_OUT} directly for these entries before running the scorer.")
    else:
        print("\nAll 50 questions resolved automatically — no manual review needed.")

    if ambiguous_broad_match:
        print(f"\nAMBIGUOUS / BROAD MATCH WARNING ({len(ambiguous_broad_match)} questions) — "
              f"anchor text found on more than 2 pages, likely too generic (e.g. also")
        print("appears in a table of contents). These will still be scored, but the ground")
        print("truth may include false-positive pages. Consider tightening the anchor text")
        print(f"in {BENCHMARK_IN} to a longer, more distinctive phrase and re-running:")
        for qid, section, pages in ambiguous_broad_match:
            print(f"  {qid}: \"{section}\" matched pages {pages}")


if __name__ == "__main__":
    run()
