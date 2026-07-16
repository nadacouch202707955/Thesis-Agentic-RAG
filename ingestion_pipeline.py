"""
Document Ingestion Pipeline — Agentic RAG Thesis
Nada Ali Yaqoob · 202507955 · Polytechnic of Bahrain

Creates three Azure AI Search indexes from documents in Blob Storage:
  kb-256   → chunk size 256 tokens
  kb-512   → chunk size 512 tokens
  kb-1024  → chunk size 1024 tokens

Run: python ingestion_pipeline.py
"""

import os
import time
import json
import re
import fitz                          # PyMuPDF
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType,
    SimpleField, SearchableField, VectorSearch,
    HnswAlgorithmConfiguration, VectorSearchProfile,
    SearchIndex
)
from azure.core.credentials import AzureKeyCredential
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
import tiktoken

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION  (values come from .env)
# ─────────────────────────────────────────────
STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
STORAGE_CONTAINER         = os.getenv("AZURE_STORAGE_CONTAINER", "academic-advising-docs")

SEARCH_ENDPOINT           = os.getenv("AZURE_SEARCH_ENDPOINT")   # https://academic-advisor-search.search.windows.net
SEARCH_ADMIN_KEY          = os.getenv("AZURE_SEARCH_ADMIN_KEY")

OPENAI_ENDPOINT           = os.getenv("AZURE_OPENAI_ENDPOINT")   # https://foundry-nada-rag.openai.azure.com
OPENAI_API_KEY            = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT      = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
EMBEDDING_DIMENSIONS      = 1536

# Three chunk configurations for RQ2 experiments
CHUNK_CONFIGS = [
    {"name": "kb-256",  "chunk_tokens": 256,  "overlap_tokens": 51},   # ~20% overlap
    {"name": "kb-512",  "chunk_tokens": 512,  "overlap_tokens": 102},
    {"name": "kb-1024", "chunk_tokens": 1024, "overlap_tokens": 205},
]

# Embedding batch size — keeps API costs low
EMBED_BATCH_SIZE = 16

# ─────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────
def get_clients():
    blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    search_index_client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=AzureKeyCredential(SEARCH_ADMIN_KEY)
    )
    openai_client = AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version="2024-02-01"
    )
    return blob_service, search_index_client, openai_client


# ─────────────────────────────────────────────
# STEP 1 — LIST ALL PDFs IN BLOB STORAGE
# ─────────────────────────────────────────────
def list_pdf_blobs(blob_service: BlobServiceClient) -> list[str]:
    container = blob_service.get_container_client(STORAGE_CONTAINER)
    blobs = [b.name for b in container.list_blobs() if b.name.lower().endswith(".pdf")]
    print(f"[Blob] Found {len(blobs)} PDF documents in '{STORAGE_CONTAINER}'")
    return blobs


# ─────────────────────────────────────────────
# STEP 2 — DOWNLOAD PDF AND EXTRACT TEXT
# ─────────────────────────────────────────────
def download_and_extract(blob_service: BlobServiceClient, blob_name: str) -> list[dict]:
    """
    Download PDF from Blob Storage and extract text page by page.
    Returns list of {page_number, text, source_document}
    """
    container = blob_service.get_container_client(STORAGE_CONTAINER)
    blob_bytes = container.download_blob(blob_name).readall()

    pages = []
    doc = fitz.open(stream=blob_bytes, filetype="pdf")
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:                          # skip blank pages
            pages.append({
                "page_number": page_num + 1,
                "text": text,
                "source_document": os.path.basename(blob_name)
            })
    doc.close()
    return pages


# ─────────────────────────────────────────────
# STEP 3 — CHUNK TEXT (LangChain RecursiveCharacterTextSplitter)
#           Using tiktoken for accurate token counting (Chapter 3 §3.4.1)
# ─────────────────────────────────────────────
def chunk_pages(pages: list[dict], chunk_tokens: int, overlap_tokens: int) -> list[dict]:
    """
    Chunks extracted page text using RecursiveCharacterTextSplitter
    with tiktoken-based token counting as described in Chapter 3 §3.4.1.
    """
    enc = tiktoken.get_encoding("cl100k_base")   # encoding used by GPT-4o

    def token_length(text: str) -> int:
        return len(enc.encode(text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        length_function=token_length,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split_text in enumerate(splits):
            chunks.append({
                "text": split_text.strip(),
                "source_document": page["source_document"],
                "source_page": page["page_number"],
                "chunk_index": i
            })

    return [c for c in chunks if len(c["text"]) > 50]   # drop very short fragments


# ─────────────────────────────────────────────
# STEP 4 — GENERATE EMBEDDINGS (batched)
#           text-embedding-3-small — fixed across all 3 KBs (Chapter 3 §3.4.2)
# ─────────────────────────────────────────────
def generate_embeddings(openai_client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings in batches to stay within rate limits.
    Embedding model is FIXED at text-embedding-3-small across all experiments
    (Chapter 3 Table 3.5 — never change this between KB-256/512/1024).
    """
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i: i + EMBED_BATCH_SIZE]
        response = openai_client.embeddings.create(
            input=batch,
            model=EMBEDDING_DEPLOYMENT
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        # polite rate-limit pause
        if i + EMBED_BATCH_SIZE < len(texts):
            time.sleep(0.5)

    return all_embeddings


# ─────────────────────────────────────────────
# STEP 5 — CREATE AZURE AI SEARCH INDEX
# ─────────────────────────────────────────────
def create_search_index(index_client: SearchIndexClient, index_name: str):
    """
    Creates the Azure AI Search index with vector + keyword fields.
    Hybrid retrieval (BM25 + vector) as per Chapter 3 §3.4.3.
    No semantic ranker — Basic tier limitation (noted in Chapter 3 §3.4.3).
    """
    # Delete existing index if it exists (fresh rebuild)
    try:
        index_client.delete_index(index_name)
        print(f"[Index] Deleted existing index '{index_name}'")
        time.sleep(2)
    except Exception:
        pass

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name="en.microsoft"    # English BM25 analyser
        ),
        SimpleField(
            name="source_document",
            type=SearchFieldDataType.String,
            filterable=True,
            retrievable=True
        ),
        SimpleField(
            name="source_page",
            type=SearchFieldDataType.Int32,
            filterable=True,
            retrievable=True
        ),
        SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            retrievable=True
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="hnsw-profile"
        )
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine"
                }
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-config"
            )
        ]
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )

    index_client.create_or_update_index(index)
    print(f"[Index] Created index '{index_name}'")


# ─────────────────────────────────────────────
# STEP 6 — UPLOAD DOCUMENTS TO INDEX (batched)
# ─────────────────────────────────────────────
def upload_to_index(
    search_index_client: SearchIndexClient,
    index_name: str,
    chunks: list[dict],
    embeddings: list[list[float]]
):
    """
    Uploads chunks + vectors to Azure AI Search in batches of 100.
    """
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=index_name,
        credential=AzureKeyCredential(SEARCH_ADMIN_KEY)
    )

    documents = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # sanitise source name for use in document ID
        safe_source = re.sub(r'[^a-zA-Z0-9_\-=]', '_', chunk["source_document"])
        documents.append({
            "id": f"{safe_source}_p{chunk['source_page']}_c{chunk['chunk_index']}_{i}",
            "content": chunk["text"],
            "source_document": chunk["source_document"],
            "source_page": chunk["source_page"],
            "chunk_index": chunk["chunk_index"],
            "content_vector": embedding
        })

    # Upload in batches of 100 (Azure Search limit per request)
    batch_size = 100
    total_uploaded = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i: i + batch_size]
        result = search_client.upload_documents(documents=batch)
        total_uploaded += len([r for r in result if r.succeeded])
        time.sleep(0.3)

    print(f"[Upload] Uploaded {total_uploaded}/{len(documents)} documents to '{index_name}'")
    return total_uploaded


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline():
    print("=" * 60)
    print("Agentic RAG — Document Ingestion Pipeline")
    print("Chapter 3 §3.4.1 — Three chunk sizes for RQ2")
    print("=" * 60)

    blob_service, search_index_client, openai_client = get_clients()

    # Get all PDF names from Blob Storage
    pdf_blobs = list_pdf_blobs(blob_service)
    if not pdf_blobs:
        print("[ERROR] No PDFs found. Check container name in .env")
        return

    # Extract text from all PDFs once (shared across all 3 KBs)
    print(f"\n[Step 1] Extracting text from {len(pdf_blobs)} PDFs...")
    all_pages = []
    for i, blob_name in enumerate(pdf_blobs):
        pages = download_and_extract(blob_service, blob_name)
        all_pages.extend(pages)
        print(f"  [{i+1}/{len(pdf_blobs)}] {blob_name} → {len(pages)} pages")

    print(f"\n[Step 1 Done] Total pages extracted: {len(all_pages)}")

    # Summary log
    summary = []

    # Run pipeline for each chunk configuration
    for config in CHUNK_CONFIGS:
        index_name    = config["name"]
        chunk_tokens  = config["chunk_tokens"]
        overlap_tokens = config["overlap_tokens"]

        print(f"\n{'─'*60}")
        print(f"[Pipeline] Building {index_name} (chunk={chunk_tokens} tokens, overlap={overlap_tokens})")
        print(f"{'─'*60}")

        # Step 2 — Chunk
        print(f"[Step 2] Chunking {len(all_pages)} pages...")
        chunks = chunk_pages(all_pages, chunk_tokens, overlap_tokens)
        print(f"[Step 2 Done] {len(chunks)} chunks created")

        # Step 3 — Embed (text-embedding-3-small, FIXED — never change)
        print(f"[Step 3] Generating embeddings for {len(chunks)} chunks...")
        print(f"         Using: {EMBEDDING_DEPLOYMENT} (fixed per Chapter 3 Table 3.5)")
        texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(openai_client, texts)
        print(f"[Step 3 Done] {len(embeddings)} embeddings generated")

        # Step 4 — Create index
        print(f"[Step 4] Creating Azure AI Search index '{index_name}'...")
        create_search_index(search_index_client, index_name)

        # Step 5 — Upload
        print(f"[Step 5] Uploading to '{index_name}'...")
        uploaded = upload_to_index(search_index_client, index_name, chunks, embeddings)

        summary.append({
            "index": index_name,
            "chunk_tokens": chunk_tokens,
            "overlap_tokens": overlap_tokens,
            "total_pages": len(all_pages),
            "total_chunks": len(chunks),
            "uploaded": uploaded
        })

        print(f"[Done] {index_name} complete.")

    # Final summary
    print(f"\n{'='*60}")
    print("INGESTION COMPLETE — Summary")
    print(f"{'='*60}")
    print(f"{'Index':<12} {'Chunk':<8} {'Overlap':<10} {'Chunks':<10} {'Uploaded'}")
    print(f"{'─'*50}")
    for s in summary:
        print(f"{s['index']:<12} {s['chunk_tokens']:<8} {s['overlap_tokens']:<10} {s['total_chunks']:<10} {s['uploaded']}")

    # Save summary for thesis records
    with open("ingestion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[Saved] ingestion_summary.json — keep this for Chapter 4")


if __name__ == "__main__":
    run_pipeline()