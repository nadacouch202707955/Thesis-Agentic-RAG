"""
Basic RAG — Baseline 2
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

Conventional RAG: query → hybrid search (BM25 + vector) → top-5 chunks → GPT-4o response
No agents. No student profile. No HITL.
This is the comparison baseline for RQ1 (Chapter 3 Table 3.7 — Condition 2).

Run: python basic_rag.py
"""

import os
import json
import time
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SEARCH_ENDPOINT      = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_ADMIN_KEY     = os.getenv("AZURE_SEARCH_ADMIN_KEY")
SEARCH_INDEX         = os.getenv("AZURE_SEARCH_INDEX", "kb-512")  # frozen after RQ2 — change to winner

OPENAI_ENDPOINT      = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY       = os.getenv("AZURE_OPENAI_API_KEY")
CHAT_DEPLOYMENT      = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini")   # switch to gpt-4o for eval
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

TOP_K = 5   # Chapter 3 Table 3.6 — fixed at 5

# ─────────────────────────────────────────────
# SYSTEM PROMPT (Chapter 3 §3.4.4)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional academic advising assistant for the Polytechnic of Bahrain.

STRICT RULES:
1. Answer ONLY using the information provided in the retrieved document chunks below.
2. ALWAYS cite your source at the end of your answer using this format:
   [Source: <document name>, Page <number>]
3. If the retrieved chunks do not contain enough information to answer the question,
   respond exactly with:
   "I don't have sufficient information in the available documents to answer this question.
   Please consult your academic advisor directly."
4. NEVER guess, invent, or assume policy details not present in the retrieved chunks.
5. Be concise, professional, and helpful.
6. If the question involves personal student data (GPA, completed courses, eligibility),
   note that you do not have access to individual student records in this mode.
"""

# ─────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────
def get_clients():
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX,
        credential=AzureKeyCredential(SEARCH_ADMIN_KEY)
    )
    openai_client = AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version="2024-02-01"
    )
    return search_client, openai_client


# ─────────────────────────────────────────────
# STEP 1 — GENERATE QUERY EMBEDDING
# ─────────────────────────────────────────────
def embed_query(openai_client: AzureOpenAI, query: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=query,
        model=EMBEDDING_DEPLOYMENT
    )
    return response.data[0].embedding


# ─────────────────────────────────────────────
# STEP 2 — HYBRID RETRIEVAL (BM25 + Vector)
#           Chapter 3 §3.4.3 — no semantic reranker (Basic tier limitation)
# ─────────────────────────────────────────────
def hybrid_search(search_client: SearchClient, query: str, query_vector: list[float]) -> list[dict]:
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=TOP_K,
        fields="content_vector"
    )

    results = search_client.search(
        search_text=query,           # BM25 keyword search
        vector_queries=[vector_query],  # dense vector search
        select=["content", "source_document", "source_page"],
        top=TOP_K
    )

    chunks = []
    for result in results:
        chunks.append({
            "content": result["content"],
            "source_document": result["source_document"],
            "source_page": result["source_page"],
            "score": result.get("@search.score", 0)
        })

    return chunks


# ─────────────────────────────────────────────
# STEP 3 — BUILD CONTEXT FROM RETRIEVED CHUNKS
# ─────────────────────────────────────────────
def build_context(chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Chunk {i+1}] Source: {chunk['source_document']}, Page {chunk['source_page']}\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(context_parts)


# ─────────────────────────────────────────────
# STEP 4 — GENERATE RESPONSE WITH GPT
# ─────────────────────────────────────────────
def generate_response(
    openai_client: AzureOpenAI,
    query: str,
    context: str,
    conversation_history: list[dict]
) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history for multi-turn support
    messages.extend(conversation_history)

    # Add current query with retrieved context
    user_message = f"""Retrieved document chunks:
{context}

Student question: {query}"""

    messages.append({"role": "user", "content": user_message})

    response = openai_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=messages,
        temperature=1,    # gpt-5-mini reasoning models only support temperature=1
        max_completion_tokens=3000   # raised from 800 — gpt-5-mini spends hidden
                                       # "reasoning tokens" from this same budget before
                                       # writing the visible answer, so a low limit can
                                       # get fully consumed by reasoning alone, leaving
                                       # an empty response with finish_reason="length"
    )

    content = response.choices[0].message.content
    print(f"[DEBUG] Raw response: {repr(content)}")
    print(f"[DEBUG] Finish reason: {response.choices[0].finish_reason}")
    if hasattr(response, "usage") and response.usage:
        print(f"[DEBUG] Token usage: {response.usage}")
    if not content or content.strip() == "":
        return "I was unable to generate a response. Please try rephrasing your question."
    return content


# ─────────────────────────────────────────────
# FULL RAG PIPELINE — single query
# ─────────────────────────────────────────────
def rag_query(
    search_client: SearchClient,
    openai_client: AzureOpenAI,
    query: str,
    conversation_history: list[dict] = [],
    verbose: bool = False
) -> dict:
    """
    Runs the full Basic RAG pipeline for one query.
    Returns dict with response, retrieved chunks, and metadata.
    Used by both the interactive chat and the RQ1 evaluation script.
    """
    start_time = time.time()

    # Step 1 — embed query
    query_vector = embed_query(openai_client, query)

    # Step 2 — hybrid retrieval
    chunks = hybrid_search(search_client, query, query_vector)

    if verbose:
        print(f"\n[Retrieval] Found {len(chunks)} chunks:")
        for i, c in enumerate(chunks):
            print(f"  [{i+1}] {c['source_document']} p.{c['source_page']} (score: {c['score']:.3f})")

    # Step 3 — build context
    context = build_context(chunks)

    # Step 4 — generate response
    response = generate_response(openai_client, query, context, conversation_history)

    elapsed = time.time() - start_time

    return {
        "query": query,
        "response": response,
        "retrieved_chunks": chunks,
        "context": context,
        "latency_seconds": round(elapsed, 2),
        "index_used": SEARCH_INDEX,
        "model_used": CHAT_DEPLOYMENT
    }


# ─────────────────────────────────────────────
# INTERACTIVE CHAT MODE
# ─────────────────────────────────────────────
def run_chat():
    print("=" * 60)
    print("Basic RAG — Academic Advising System")
    print("Polytechnic of Bahrain | Baseline 2")
    print(f"Index: {SEARCH_INDEX} | Model: {CHAT_DEPLOYMENT}")
    print("=" * 60)
    print("Type your question. Type 'quit' to exit.\n")

    search_client, openai_client = get_clients()
    conversation_history = []

    while True:
        query = input("Student: ").strip()
        if not query:
            continue
        if query.lower() in ["quit", "exit", "q"]:
            print("Session ended.")
            break

        result = rag_query(
            search_client, openai_client, query,
            conversation_history, verbose=True
        )

        print(f"\nAdvisor: {result['response']}")
        print(f"[Latency: {result['latency_seconds']}s]\n")

        # Update conversation history for multi-turn
        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": result["response"]})

        # Keep only last 6 turns to avoid context overflow
        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]


# ─────────────────────────────────────────────
# BATCH EVALUATION MODE (used by RQ1 eval script)
# ─────────────────────────────────────────────
def run_batch_evaluation(questions_file: str, output_file: str):
    """
    Runs all questions from a JSON file through Basic RAG.
    Saves results for RAGAS evaluation.
    """
    print(f"[Batch] Loading questions from {questions_file}")
    with open(questions_file, "r") as f:
        questions = json.load(f)

    search_client, openai_client = get_clients()
    results = []

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['question'][:60]}...")
        result = rag_query(search_client, openai_client, q["question"])

        results.append({
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q["ground_truth_answer"],
            "answer": result["response"],
            "contexts": [c["content"] for c in result["retrieved_chunks"]],
            "source_documents": [c["source_document"] for c in result["retrieved_chunks"]],
            "latency_seconds": result["latency_seconds"]
        })

        time.sleep(0.5)   # rate limit pause

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Batch Done] Results saved to {output_file}")
    return results


if __name__ == "__main__":
    run_chat()