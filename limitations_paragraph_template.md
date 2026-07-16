# Limitations paragraph — drop into Chapter 4.3 or 4.4

Use/adapt this after you have your actual numbers from p5_report.json.
Placeholders in [brackets] — fill in from your results.

---

While kb-[BEST_CONFIG] achieved the highest average Precision@5 (M=[X.XX],
SD=[X.XX]) across the 50-question benchmark and was therefore selected as
the frozen retrieval configuration for the subsequent Agentic RAG system
(Section 4.5), this result should be interpreted with several caveats.
First, the benchmark comprises 50 researcher-authored questions phrased in
clean, well-formed academic English closely mirroring the source documents'
own terminology; real student queries may include typos, informal phrasing,
or ambiguous multi-topic questions not represented in this set, and
retrieval robustness to such phrasing drift was not evaluated. Second,
performance was not uniform across question categories: [kb-CONFIG] scored
highest on [category] questions (M=[X.XX]) but was outperformed by
[other-config] on [category] questions (M=[X.XX]), suggesting that no single
chunking configuration is uniformly optimal across all query types an
academic advisor might encounter. Third, with only 50 questions per
configuration, the observed differences between chunk sizes (Table [X])
should be treated as indicative rather than statistically definitive; a
larger and more diverse benchmark, together with formal significance
testing between configurations, would strengthen confidence in the
generalisability of this result to unseen student queries post-deployment.
Manual inspection of the ten lowest-scoring questions per configuration
(Appendix [X]) was conducted to characterise failure modes — see Section
4.3.[X] for the resulting error analysis.

---

## Error analysis section template (pairs with the paragraph above)

For each config's worst_10_for_error_analysis list in p5_report.json,
categorise each failure into one of:

- **Chunk boundary cut**: the answer was split across two chunks, so no
  single retrieved chunk contained the complete relevant section
- **Wrong section retrieved**: the query matched semantically similar but
  incorrect content (e.g. a different policy section using similar wording)
- **Question ambiguity**: the question could plausibly map to more than one
  section (flag these as benchmark design issues, not retrieval failures)
- **Sparse/short source content**: the ground-truth section is very short
  (e.g. a single contact email), making it easy for the retriever to be
  outcompeted by longer, topically-adjacent chunks

Report the count of each failure type per configuration in a short table —
this is the qualitative complement to the P@5 numbers and is usually what
a viva panel asks about first ("why did retrieval fail here?"), so having
it ready strengthens your defense of the frozen config decision.
