"""
agentic_rag.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

Proposed System — Agentic RAG Evaluation Wrapper
=================================================
Chapter 3, Table 3.7 — Condition 4

PURPOSE
-------
This file serves as the evaluation interface for the proposed Agentic RAG
system within the RQ1 assessment pipeline. It imports run_agentic_query()
from orchestrator_agent.py and exposes a standardised run_agentic_rag_query()
function that ragas_eval_rq1.py calls uniformly across all four systems.

For each of the 50 benchmark questions, the function invokes the full
five-agent pipeline — Orchestrator, Retrieval, Profile, Validator, and
Notification — and returns the generated answer alongside the retrieved
document chunks. These outputs are passed to the RAGAS framework to compute
Faithfulness and Answer Relevance scores, which together answer RQ1 by
measuring whether the Agentic RAG system produces more accurate and relevant
academic advising responses than the three baseline systems.

Why a wrapper?
--------------
orchestrator_agent.py is a full interactive chat system with conversation
history, verbose terminal output, and multi-turn session management.
ragas_eval_rq1.py needs a simple, clean interface: give it a question,
get back an answer and retrieved chunks. This wrapper adapts the full
pipeline to the standard interface expected by the evaluation script.

Usage (batch evaluation — called by ragas_eval_rq1.py):
    from agentic_rag import run_agentic_rag_query

Usage (quick test):
    py agentic_rag.py
"""

import time
from orchestrator_agent import run_agentic_query


def run_agentic_rag_query(query: str,
                           student_id: str = None,
                           verbose: bool = False) -> dict:
    """
    Evaluation wrapper for the Agentic RAG system (Condition 4).

    Calls the full five-agent pipeline from orchestrator_agent.py
    and returns results in the standard format expected by
    ragas_eval_rq1.py for RAGAS scoring.

    Parameters:
        query      — student academic advising question (string)
        student_id — optional student ID for profile personalisation
        verbose    — set True to print agent step outputs to terminal

    Returns dict with keys:
        query           — original question
        answer          — generated response from the full pipeline
        contexts        — list of retrieved chunk texts (for RAGAS)
        ground_truth    — empty string (filled by ragas_eval_rq1.py)
        system          — "agentic_rag" (system identifier)
        latency_seconds — total pipeline latency
        confidence      — Validator Agent confidence score
        escalated       — True if response was escalated to HITL
    """
    result = run_agentic_query(
        query=query,
        student_id=student_id,
        conversation_history=[],
        verbose=verbose
    )

    # Extract chunk texts for RAGAS context evaluation
    contexts = [
        chunk["content"]
        for chunk in result.get("retrieved_chunks", [])
    ]

    return {
        "query":           query,
        "answer":          result["final_response"],
        "contexts":        contexts,
        "ground_truth":    "",   # filled by ragas_eval_rq1.py
        "system":          "agentic_rag",
        "latency_seconds": result["latency_seconds"],
        "confidence":      result["validation"]["confidence_score"],
        "escalated":       result["notification"]["escalated"],
    }


if __name__ == "__main__":
    # Quick smoke test — verifies the wrapper works before RQ1 evaluation
    print("=" * 55)
    print("Agentic RAG — Evaluation Wrapper Smoke Test")
    print("=" * 55)

    test_queries = [
        "What are the academic misconduct consequences?",
        "What is the minimum GPA required to maintain good academic standing?",
        "When does semester 2 registration open?",
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n[Test {i}/3] {q[:60]}...")
        result = run_agentic_rag_query(q, verbose=False)
        print(f"  Answer ({result['latency_seconds']}s): {result['answer'][:120]}...")
        print(f"  Contexts: {len(result['contexts'])} chunks retrieved")
        print(f"  Confidence: {result['confidence']} | "
              f"Escalated: {result['escalated']}")

    print("\n✅ Smoke test complete — agentic_rag.py ready for ragas_eval_rq1.py")
