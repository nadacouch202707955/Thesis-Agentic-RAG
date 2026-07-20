"""
config_frozen.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

FROZEN RETRIEVAL CONFIGURATION — DO NOT CHANGE AFTER THIS DATE
===============================================================
Selected based on RQ2 Retrieval Precision@5 experiments (20 Jul 2026).

Scientific justification:
    Both Basic RAG (Baseline 2) and Agentic RAG (Proposed System) use
    this identical retrieval configuration for RQ1 evaluation.
    Keeping retrieval constant ensures that any performance difference
    measured by RAGAS Faithfulness and Answer Relevance is attributable
    solely to the agentic architecture, not to retrieval variation.
    This is the core scientific control required by RQ1.

RQ2 Precision@5 Results (50 benchmark questions):
    kb-256  → mean P@5 = 0.528  (sd = 0.328,  n = 50,  chunks = 1,271)
    kb-512  → mean P@5 = 0.544  (sd = 0.355,  n = 50,  chunks = 689)   ← WINNER
    kb-1024 → mean P@5 = 0.536  (sd = 0.319,  n = 50,  chunks = 438)

Selection date : 20 July 2026
Scorer script  : p5_scorer_rq2.py
Full report    : p5_report_rq2.json
CSV appendix   : p5_results_rq2.csv
"""

FROZEN_CONFIG = {
    # ── Knowledge Base ─────────────────────────────────────────────
    "index_name":           "kb-512",
    "chunk_size_tokens":    512,
    "overlap_tokens":       102,        # ~20% overlap

    # ── Embedding model (FIXED throughout all experiments) ─────────
    "embedding_model":      "text-embedding-3-small",
    "embedding_dimensions": 1536,

    # ── Retrieval strategy ─────────────────────────────────────────
    "retrieval_mode":       "hybrid",   # BM25 keyword + dense vector
    "fusion_method":        "RRF",      # Reciprocal Rank Fusion
    "top_k":                5,          # top-5 chunks passed to GPT

    # ── Scoring metadata ───────────────────────────────────────────
    "page_tolerance":       1,          # ±1 page window for P@5 scoring
    "rq2_mean_p5":          0.544,
    "rq2_sd":               0.355,
    "n_questions_scored":   50,

    # ── Selection record ───────────────────────────────────────────
    "selected_date":        "2026-07-20",
    "selected_by":          "Nada Ali Yaqoob · 202507955",
    "note": (
        "DO NOT modify this file after 20 Jul 2026. "
        "All RQ1 evaluation runs (Basic RAG, Single-Agent RAG, "
        "Agentic RAG) must use index_name=kb-512 and top_k=5."
    ),
}


def get_frozen_config() -> dict:
    """
    Returns the frozen retrieval configuration dictionary.
    Import and call this function in every RAG script to ensure
    the same retrieval settings are used across all four systems.

    Usage:
        from config_frozen import get_frozen_config
        config = get_frozen_config()
        index  = config["index_name"]   # "kb-512"
        top_k  = config["top_k"]        # 5
    """
    return FROZEN_CONFIG.copy()


if __name__ == "__main__":
    import json
    print("=== FROZEN RETRIEVAL CONFIGURATION ===")
    print(json.dumps(FROZEN_CONFIG, indent=2))
    print("\nThis configuration is LOCKED for RQ1 evaluation.")
    print(f"Winner: {FROZEN_CONFIG['index_name']}  "
          f"(mean P@5 = {FROZEN_CONFIG['rq2_mean_p5']})")
