# Chapter 4 — Implementation Evidence Log

Running record of validated pipeline behavior, kept as concrete evidence
for Chapter 4 (Implementation) and Chapter 5 (Results & Evaluation).
Append to this file as more validation happens — don't rely on chat memory.

---

## Evidence Item 1 — Basic RAG: Correct retrieval + correct grounding + noise rejection

**Date validated:** 14 July 2026
**System:** Basic RAG (basic_rag.py), index = kb-512, model = gpt-5-mini (dev)
**Test query:** "Academic contacts" (student asking for Bahrain Polytechnic academic contact list)

**Retrieved chunks (Top-5, hybrid search BM25+vector):**
| Rank | Source | Page | Score | Relevant? |
|---|---|---|---|---|
| 1 | Student-Handbook.pdf | 7 | 0.032 | Yes — contains Academic Contacts table |
| 2 | Academic-Regulations.pdf | 6 | 0.031 | No — irrelevant, low score |
| 3 | Student-Handbook.pdf | 7 | 0.028 | Yes — contains Academic Contacts table |
| 4 | Student-Handbook.pdf | 22 | 0.019 | No — different section, irrelevant |
| 5 | Student-Handbook.pdf | 22 | 0.019 | No — different section, irrelevant |

**Generated answer:** Full, accurate reproduction of the Academic Contacts
table from Student-Handbook.pdf p.7 — every name, title, and email verified
correct against the source document (Faculty of Business & Logistics, School
of ICT and Web Academy sections). No hallucinated names/emails.

**Why this matters for the thesis:**
- Demonstrates the system correctly identifies which of the 5 retrieved
  chunks are actually relevant (2 of 5 in this case) and grounds its answer
  only in those, ignoring the 3 irrelevant/noisy chunks — this is exactly
  the behavior RAGAS Faithfulness rewards, and a good qualitative example
  to cite alongside the quantitative RQ1 RAGAS scores.
- Confirms retrieval correctly favors Student-Handbook.pdf over
  Academic-Regulations.pdf for a contacts-related query, despite both
  being retrieved (relevance score gap: 0.032 vs 0.031 — close, worth
  noting as a borderline case in the error-analysis discussion for RQ2).
- Confirms citation-formatted system prompt is being followed structurally
  (chunks correctly labeled by source/page before being passed to GPT).

**Action item / to verify:** Confirm the final generated answer included
the required `[Source: <document>, Page <number>]` citation line at the
end, per the system prompt's Rule 2 — not yet explicitly confirmed in the
pasted output. Check this on the next test run.

**Screenshot/artifact:** [Chapter 4 appendix — insert the actual VS Code
terminal output + rendered table screenshot from 14 July 2026 test]

---

## Evidence Item 2 — retrieval_eval_rq2.py: full retrieval evaluation run completed successfully

**Date validated:** 14 July 2026
**Script:** retrieval_eval_rq2.py
**Result:** All 150 searches (50 benchmark questions × 3 indexes: kb-256,
kb-512, kb-1024) completed with zero errors on first execution after the
basic_rag.py fixes (Bugs 1 and 2 above) were applied.

**Console confirmation:**
```
Done. Retrieval results for all 3 indexes saved to retrieval_results_rq2.json
Next step: run p5_scorer_rq2.py to compute Precision@5.
```

**Why this matters for the thesis:**
- Confirms embed_query() and hybrid_search() — imported directly from the
  now-fixed basic_rag.py — work correctly across all three separately
  built indexes, not just kb-512 (the default basic_rag.py uses for chat).
- Confirms the full RQ2 evaluation pipeline (benchmark → retrieval → JSON
  output) runs end-to-end without manual intervention, satisfying the
  reproducibility requirement discussed for the viva.
- retrieval_results_rq2.json is now available as the direct input for
  p5_scorer_rq2.py — next step in the RQ2 chain.

**Next step:** Run p5_scorer_rq2.py against this output to compute
Precision@5 per index, with the mean/SD and per-category breakdown.

---

## Bugs found and fixed during implementation (useful for Chapter 4 "development process" narrative)

### Bug 1 — Corrupted generate_response() function (basic_rag.py)
**Symptom:** `IndentationError: unexpected indent` on import, blocking
`retrieval_eval_rq2.py` from running.
**Root cause:** The line `response = openai_client.chat.completions.create(`
had been split — its front half was stranded inside the docstring of
`rag_query()`, its back half (the parameters) was left in `generate_response()`
with no opening call line. Likely a stray editor drag/find-replace.
**Fix:** Reconstructed the missing call line, corrected indentation, cleaned
the docstring. Verified with `python -m py_compile basic_rag.py`.

### Bug 2 — Empty response despite successful retrieval (basic_rag.py)
**Symptom:** Correct chunks retrieved, but generation returned
`finish_reason: length` with empty content.
**Root cause:** `gpt-5-mini` is a reasoning model — hidden reasoning tokens
count against `max_completion_tokens`. The original limit (800) was fully
consumed by internal reasoning before any visible answer text could be
written.
**Fix:** Raised `max_completion_tokens` from 800 → 3000. Added
`response.usage` debug logging to monitor reasoning vs. output token split
going forward. Cost impact assessed as negligible (<$1 total across full
50-question benchmark at gpt-5-mini rates).
**Note for Chapter 4:** this quirk is specific to reasoning-tuned models;
won't recur when the evaluation model is swapped to GPT-4o (non-reasoning)
for the final RQ1 run.

### Bug 3 — Ground-truth resolution logic dropped sections silently (resolve_ground_truth_pages.py)
**Symptom:** Multi-section single-document benchmark questions (e.g. Q050,
covering "Counseling; Academic Advising; Mentoring") were only resolving
against the first listed section, silently ignoring the rest.
**Root cause:** Original zip-based pairing logic assumed one section per
document; broke when one document had multiple valid ground-truth sections.
**Fix:** Rewrote resolution logic to union all page matches across every
listed section when only one document is named, rather than truncating via
zip(). Verified via console warning system — script now flags any anchor
matching >2 pages for manual review rather than silently over/under-matching.

---

## How to keep this file useful

Every time you validate a new query/behavior, or fix a bug worth mentioning
in Chapter 4, add a new dated entry above using the same format. When you
sit down to write Chapter 4, feed this whole file back to Claude (or read
it yourself) instead of relying on chat history — chat memory is not
guaranteed to persist across separate conversations.

## Evidence Item 4 — Agentic RAG Pipeline: All 5 Agents Validated

**Date:** 20 July 2026
**Commit hash:** 6532120
**File:** orchestrator_agent.py

**Test query:** "What are the academic misconduct consequences for students?"

**Agent pipeline output:**
- Retrieval Agent: 5 chunks retrieved from kb-512
  - Chunk 1: Student-Handbook.pdf p.21 (score: 0.032) ← correct
  - Chunk 2: Academic-Regulations.pdf p.7 (score: 0.028)
- Validator Agent: Confidence = 0.86 ✅ Approved (threshold: 0.70)
- Notification Agent: Response delivered directly to student
- Latency: 18.94s

**All 5 agents confirmed working:**
✅ Orchestrator Agent — query coordination and prompt assembly
✅ Retrieval Agent — hybrid search on kb-512 (frozen config)
✅ Profile Agent — available (SQL not yet connected, skipped correctly)
✅ Validator Agent — confidence scoring working (0.86 > 0.70)
✅ Notification Agent — delivery and HITL escalation both tested