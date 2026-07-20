"""
orchestrator_agent.py
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

Orchestrator Agent — central coordinator of the Agentic RAG pipeline.

Workflow:
  1. Receive student query + optional student_id
  2. Call Retrieval Agent  → top-5 document chunks (frozen config)
  3. Call Profile Agent    → student academic profile from Azure SQL
  4. Assemble prompt       → system + retrieved context + profile + query
  5. Call GPT              → generate grounded response
  6. Call Validator Agent  → confidence check
  7. Call Notification Agent → deliver or escalate via HITL

This file is the main entry point for the Agentic RAG system (Proposed System,
Chapter 3 Table 3.7 Condition 4). It is imported by ragas_eval_rq1.py for
the formal RQ1 RAGAS evaluation.

Usage (interactive):
    py orchestrator_agent.py

Usage (batch evaluation):
    from orchestrator_agent import run_agentic_query
"""

import os
import json
import time
from dotenv import load_dotenv
from openai import AzureOpenAI

# Import frozen retrieval config (RQ2 winner — DO NOT CHANGE)
from config_frozen import get_frozen_config

# Import retrieval functions from Basic RAG (reuse — keeps config identical)
from basic_rag import embed_query, hybrid_search, get_clients as get_rag_clients

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
FROZEN = get_frozen_config()

OPENAI_ENDPOINT      = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY       = os.getenv("AZURE_OPENAI_API_KEY")
CHAT_DEPLOYMENT      = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini")
CONFIDENCE_THRESHOLD = 0.70   # below this → HITL escalation

# ─────────────────────────────────────────────
# SYSTEM PROMPT — Agentic RAG (Chapter 3 §3.4.4)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional academic advising assistant for a higher education institution.
You have access to official institutional policy documents and the student's academic profile.

STRICT RULES:
1. Answer ONLY using the retrieved document chunks and student profile provided below.
2. ALWAYS cite your source: [Source: <document name>, Page <number>]
3. If the retrieved context is insufficient, respond:
   "I don't have sufficient information in the available documents to answer this.
   Please consult your academic advisor directly."
4. NEVER guess, invent, or assume policy details not present in the retrieved chunks.
5. When student profile data is available, personalise your response accordingly.
6. Be concise, accurate, and professional."""


# ─────────────────────────────────────────────
# RETRIEVAL AGENT
# Returns top-5 chunks using FROZEN config
# ─────────────────────────────────────────────
def retrieval_agent(openai_client: AzureOpenAI,
                    search_client,
                    query: str) -> list[dict]:
    """
    Retrieval Agent — executes hybrid search on kb-512 (frozen config).
    Returns list of top-5 document chunks with source metadata.
    Imports embed_query and hybrid_search from basic_rag.py to ensure
    identical retrieval behaviour across all 4 evaluation systems.
    """
    query_vector = embed_query(openai_client, query)
    chunks = hybrid_search(search_client, query, query_vector)

    print(f"  [Retrieval Agent] Retrieved {len(chunks)} chunks from {FROZEN['index_name']}")
    for i, c in enumerate(chunks):
        print(f"    [{i+1}] {c['source_document']} p.{c['source_page']} "
              f"(score: {c.get('score', 0):.3f})")
    return chunks


# ─────────────────────────────────────────────
# PROFILE AGENT
# Queries Azure SQL for student academic profile
# ─────────────────────────────────────────────
def profile_agent(student_id: str = None) -> dict:
    """
    Profile Agent — retrieves student academic profile from Azure SQL Database.
    Returns structured profile dict for personalised advising.
    If student_id is None or SQL unavailable, returns empty profile.
    """
    if not student_id:
        return {}

    try:
        import pyodbc
        conn_str = os.getenv("AZURE_SQL_CONNECTION_STRING")
        if not conn_str:
            print("  [Profile Agent] No SQL connection string — skipping profile")
            return {}

        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                s.student_id,
                s.first_name,
                s.last_name,
                s.gpa,
                s.credits_completed,
                s.academic_standing,
                pl.programme_name,
                gl.gender_name,
                sl.scholarship_name
            FROM students s
            LEFT JOIN ProgrammeLookup pl ON s.programme_id   = pl.programme_id
            LEFT JOIN GenderLookup    gl ON s.gender_id      = gl.gender_id
            LEFT JOIN ScholarshipLookup sl ON s.scholarship_id = sl.scholarship_id
            WHERE s.student_id = ?
        """, student_id)

        row = cursor.fetchone()
        conn.close()

        if not row:
            print(f"  [Profile Agent] Student {student_id} not found in database")
            return {}

        profile = {
            "student_id":        row.student_id,
            "name":              f"{row.first_name} {row.last_name}",
            "programme":         row.programme_name or "Unknown",
            "gpa":               float(row.gpa) if row.gpa else None,
            "credits_completed": int(row.credits_completed) if row.credits_completed else 0,
            "academic_standing": row.academic_standing or "Unknown",
            "gender":            row.gender_name or "Unknown",
            "scholarship":       row.scholarship_name or "None",
        }

        print(f"  [Profile Agent] Retrieved profile for {profile['name']} "
              f"(GPA: {profile['gpa']}, Credits: {profile['credits_completed']})")
        return profile

    except Exception as e:
        print(f"  [Profile Agent] SQL error: {e} — continuing without profile")
        return {}


# ─────────────────────────────────────────────
# PROMPT ASSEMBLY
# Combines retrieved chunks + student profile + query
# ─────────────────────────────────────────────
def assemble_prompt(query: str,
                    chunks: list[dict],
                    profile: dict,
                    conversation_history: list[dict]) -> list[dict]:
    """
    Assembles the full message list for GPT:
      system prompt + conversation history + user message
      (user message = retrieved context + student profile + query)
    """
    # Build retrieved context block
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Chunk {i+1}] Source: {chunk['source_document']}, "
            f"Page {chunk['source_page']}\n{chunk['content']}"
        )
    context_block = "\n\n".join(context_parts)

    # Build student profile block (only if profile available)
    if profile:
        profile_block = f"""
Student Profile:
  Name:              {profile.get('name', 'Unknown')}
  Programme:         {profile.get('programme', 'Unknown')}
  GPA:               {profile.get('gpa', 'Unknown')}
  Credits Completed: {profile.get('credits_completed', 'Unknown')}
  Academic Standing: {profile.get('academic_standing', 'Unknown')}
  Scholarship:       {profile.get('scholarship', 'None')}"""
    else:
        profile_block = "Student Profile: Not available for this query."

    user_message = f"""Retrieved document chunks:
{context_block}

{profile_block}

Student question: {query}"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})
    return messages


# ─────────────────────────────────────────────
# VALIDATOR AGENT
# Checks response confidence against retrieved context
# ─────────────────────────────────────────────
def validator_agent(response: str, chunks: list[dict]) -> dict:
    """
    Validator Agent — assesses response quality and confidence.

    Confidence scoring method:
      - Checks how many key phrases from the response appear in retrieved chunks
      - Scores 1.0 if response is well-grounded in retrieved context
      - Scores lower if response contains content not traceable to chunks
      - Threshold: < 0.70 → escalate to HITL

    Note: Full RAGAS faithfulness scoring is used in the formal evaluation
    (ragas_eval_rq1.py). This lightweight check is used in real-time to
    decide whether HITL escalation is needed.
    """
    if not response or response.strip() == "":
        return {"approved": False, "confidence_score": 0.0,
                "reason": "Empty response generated"}

    # Check for known low-confidence patterns
    low_confidence_phrases = [
        "i don't have sufficient information",
        "i cannot find",
        "i am unable to",
        "please consult",
        "not available in the documents",
    ]

    response_lower = response.lower()
    for phrase in low_confidence_phrases:
        if phrase in response_lower:
            return {
                "approved": False,
                "confidence_score": 0.40,
                "reason": f"Response indicates insufficient information: '{phrase}'"
            }

    # Check grounding — how many chunks contributed to the response
    all_chunk_text = " ".join(c["content"].lower() for c in chunks)
    response_words = set(response_lower.split())
    chunk_words    = set(all_chunk_text.split())

    # Key content words from response (filter out stop words)
    stop_words = {"the","a","an","is","are","was","were","in","of","to","for",
                  "and","or","but","with","this","that","it","be","has","have",
                  "at","by","from","as","on","its","their","they","which","who"}
    content_words = [w for w in response_words if len(w) > 4 and w not in stop_words]

    if not content_words:
        confidence = 0.75
    else:
        overlap = sum(1 for w in content_words if w in chunk_words)
        confidence = min(1.0, overlap / len(content_words) + 0.3)

    approved = confidence >= CONFIDENCE_THRESHOLD

    print(f"  [Validator Agent] Confidence: {confidence:.2f} "
          f"({'✅ Approved' if approved else '⚠️ Escalate to HITL'})")

    return {
        "approved":         approved,
        "confidence_score": round(confidence, 3),
        "reason":           "Response grounded in retrieved context" if approved
                            else "Confidence below threshold — HITL escalation required"
    }


# ─────────────────────────────────────────────
# NOTIFICATION AGENT
# Delivers response or escalates to HITL
# ─────────────────────────────────────────────
def notification_agent(response: str,
                        validation: dict,
                        query: str,
                        chunks: list[dict],
                        student_id: str = None) -> dict:
    """
    Notification Agent — delivers approved responses or triggers HITL escalation.

    High confidence path: response delivered directly to student.
    Low confidence path:  escalation logged to escalations.json
                          Logic Apps HITL trigger (if configured).
    """
    if validation["approved"]:
        print("  [Notification Agent] ✅ Response approved — delivering to student")
        return {
            "delivered":        True,
            "escalated":        False,
            "final_response":   response,
            "confidence_score": validation["confidence_score"],
        }

    # HITL Escalation path
    print("  [Notification Agent] ⚠️ Low confidence — escalating to academic advisor")

    escalation_record = {
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        "student_id":       student_id or "anonymous",
        "query":            query,
        "ai_draft_answer":  response,
        "confidence_score": validation["confidence_score"],
        "reason":           validation["reason"],
        "retrieved_sources": [
            f"{c['source_document']} p.{c['source_page']}" for c in chunks
        ],
        "status":           "pending_advisor_review",
    }

    # Log escalation to file (evidence for thesis §5.4 escalation analysis)
    try:
        existing = []
        try:
            with open("escalations.json", "r") as f:
                existing = json.load(f)
        except FileNotFoundError:
            pass
        existing.append(escalation_record)
        with open("escalations.json", "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        print("  [Notification Agent] Escalation logged to escalations.json")
    except Exception as e:
        print(f"  [Notification Agent] Could not log escalation: {e}")

    # Try Logic Apps HTTP trigger (if configured)
    logic_apps_url = os.getenv("AZURE_LOGIC_APPS_URL")
    if logic_apps_url:
        try:
            import urllib.request
            import urllib.error
            payload = json.dumps(escalation_record).encode("utf-8")
            req = urllib.request.Request(
                logic_apps_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"  [Notification Agent] Logic Apps triggered: {resp.status}")
        except Exception as e:
            print(f"  [Notification Agent] Logic Apps unavailable ({e}) — "
                  f"Python fallback active")

    escalation_message = (
        "Your question has been forwarded to an academic advisor for review. "
        "You will receive a verified response shortly. "
        f"Reference ID: {escalation_record['timestamp']}"
    )

    return {
        "delivered":        False,
        "escalated":        True,
        "final_response":   escalation_message,
        "confidence_score": validation["confidence_score"],
        "escalation_record": escalation_record,
    }


# ─────────────────────────────────────────────
# MAIN PIPELINE — run_agentic_query()
# Called by interactive chat and RQ1 eval script
# ─────────────────────────────────────────────
def run_agentic_query(query: str,
                      student_id: str = None,
                      conversation_history: list = None,
                      verbose: bool = True) -> dict:
    """
    Runs the full Agentic RAG pipeline for one query.
    Returns dict with final response, all agent outputs, and metadata.
    Used by both interactive chat and ragas_eval_rq1.py batch evaluation.

    Parameters:
        query               — student's academic advising question
        student_id          — optional, for personalised profile retrieval
        conversation_history — for multi-turn context
        verbose             — print agent step outputs

    Returns:
        {
          "query":              str,
          "final_response":     str,
          "retrieved_chunks":   list,
          "student_profile":    dict,
          "validation":         dict,
          "notification":       dict,
          "latency_seconds":    float,
          "model_used":         str,
          "index_used":         str,
        }
    """
    if conversation_history is None:
        conversation_history = []

    start_time = time.time()

    if verbose:
        print(f"\n{'─'*55}")
        print(f"[Orchestrator] Query: {query[:70]}...")
        print(f"[Orchestrator] Student ID: {student_id or 'None (no personalisation)'}")

    # ── Step 1: Retrieval Agent ───────────────────────────────────
    search_client, openai_client = get_rag_clients()
    chunks = retrieval_agent(openai_client, search_client, query)

    # ── Step 2: Profile Agent ─────────────────────────────────────
    profile = profile_agent(student_id)

    # ── Step 3: Assemble prompt ───────────────────────────────────
    messages = assemble_prompt(query, chunks, profile, conversation_history)

    # ── Step 4: Generate response with GPT ───────────────────────
    if verbose:
        print(f"  [Orchestrator] Calling {CHAT_DEPLOYMENT}...")

    openai_gen = AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version="2024-02-01"
    )
    gpt_response = openai_gen.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=messages,
        max_completion_tokens=4000
    )
    raw_response = gpt_response.choices[0].message.content or ""
    raw_response = gpt_response.choices[0].message.content or ""
    print(f"  [DEBUG] GPT raw: {repr(raw_response[:300])}")

    # ── Step 5: Validator Agent ───────────────────────────────────
    validation = validator_agent(raw_response, chunks)

    # ── Step 6: Notification Agent ────────────────────────────────
    notification = notification_agent(
        raw_response, validation, query, chunks, student_id
    )

    elapsed = round(time.time() - start_time, 2)

    if verbose:
        print(f"[Orchestrator] ✅ Complete in {elapsed}s")

    return {
        "query":             query,
        "final_response":    notification["final_response"],
        "retrieved_chunks":  chunks,
        "student_profile":   profile,
        "validation":        validation,
        "notification":      notification,
        "latency_seconds":   elapsed,
        "model_used":        CHAT_DEPLOYMENT,
        "index_used":        FROZEN["index_name"],
        "context":           "\n\n".join(
            f"[Chunk {i+1}] {c['source_document']} p.{c['source_page']}\n{c['content']}"
            for i, c in enumerate(chunks)
        ),
    }


# ─────────────────────────────────────────────
# INTERACTIVE CHAT MODE
# ─────────────────────────────────────────────
def run_chat():
    print("=" * 55)
    print("Agentic RAG — Academic Advising System")
    print("Proposed System (Chapter 3 Table 3.7 — Condition 4)")
    print(f"Index: {FROZEN['index_name']} | Model: {CHAT_DEPLOYMENT}")
    print(f"Agents: Orchestrator → Retrieval → Profile → Validator → Notification")
    print("=" * 55)
    print("Type your question. Type 'quit' to exit.")
    print("Optionally type student ID after question: 'question | S12345'\n")

    conversation_history = []

    while True:
        user_input = input("Student: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Session ended.")
            break

        # Parse optional student ID (format: "question | STUDENT_ID")
        if "|" in user_input:
            parts      = user_input.split("|", 1)
            query      = parts[0].strip()
            student_id = parts[1].strip()
        else:
            query      = user_input
            student_id = None

        result = run_agentic_query(
            query, student_id, conversation_history, verbose=True
        )

        print(f"\nAdvisor: {result['final_response']}")
        if result["notification"]["escalated"]:
            print("[⚠️ This response has been flagged for advisor review]")
        print(f"[Latency: {result['latency_seconds']}s | "
              f"Confidence: {result['validation']['confidence_score']}]\n")

        # Update conversation history
        conversation_history.append({"role": "user",      "content": query})
        conversation_history.append({"role": "assistant", "content": result["final_response"]})

        # Keep last 6 turns only
        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]


if __name__ == "__main__":
    run_chat()
