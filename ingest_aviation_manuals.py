"""
Aviation Manual Ingestion Pipeline for Elasticsearch
Parses PDFs, chunks text, extracts metadata, generates embeddings, and indexes documents
"""

import os
import re
from typing import List, Dict
from uuid import uuid4

import PyPDF2
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

# Configuration
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASS = os.getenv("ES_PASS", "changeme")
INDEX_NAME = "aviation_manuals"

# Initialize Elasticsearch client
es = Elasticsearch(
    ES_HOST,
    basic_auth=(ES_USER, ES_PASS),
    verify_certs=False
)

# Initialize embedding model (384-dimensional)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def create_index():
    """
    Create Elasticsearch index with mappings for hybrid search
    Includes dense_vector field for semantic search and text fields for BM25
    """
    if es.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists")
        return
    
    es.indices.create(
        index=INDEX_NAME,
        body={
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "aviation_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "aviation_analyzer"
                    },
                    "section": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "chapter": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "part_number": {
                        "type": "keyword"
                    },
                    "manual_id": {
                        "type": "keyword"
                    },
                    "page": {
                        "type": "integer"
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
    )
    print(f"Created index '{INDEX_NAME}' with hybrid search mappings")


def extract_text_by_page(pdf_path: str) -> List[Dict]:
    """
    Extract text from PDF, page by page
    Returns list of dicts with page number and text content
    """
    docs = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()
            if text:  # Only include non-empty pages
                docs.append({"page": i, "text": text})
    return docs


def chunk_text(text: str, max_tokens: int = 800, overlap: int = 120) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum words per chunk (~800 words)
        overlap: Number of overlapping words between chunks (120 words)
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        # Move start position with overlap
        if end >= len(words):
            break
        start = end - overlap
    
    # Filter out very small fragments
    return [c for c in chunks if len(c.split()) > 50]


def infer_section(text: str) -> str:
    """
    Extract section information from text using regex patterns
    Looks for patterns like "SECTION 3.2: Engine Systems"
    """
    patterns = [
        r"SECTION\s+\d+[\.\d]*\s*[:\-]\s*[A-Z][A-Za-z0-9\-\s]+",
        r"Section\s+\d+[\.\d]*\s*[:\-]\s*[A-Z][A-Za-z0-9\-\s]+"
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return ""


def infer_chapter(text: str) -> str:
    """
    Extract ATA chapter information
    Looks for patterns like "ATA Chapter 49" or "ATA 49"
    """
    patterns = [
        r"ATA\s*Chapter\s*\d{2}",
        r"ATA\s*\d{2}"
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return ""


def infer_part_number(text: str) -> str:
    """
    Extract part numbers from text
    Looks for patterns like "APU-MSTR-RESET" or "ENG-12345-A"
    """
    m = re.search(r"\b([A-Z]{2,}-[A-Z0-9]{2,}[A-Z0-9\-]*)\b", text)
    return m.group(1) if m else ""


def index_pdf(pdf_path: str, manual_id: str):
    """
    Complete ingestion pipeline:
    1. Parse PDF by page
    2. Chunk text with overlap
    3. Extract metadata (section, chapter, part number)
    4. Generate embeddings
    5. Bulk index to Elasticsearch
    
    Args:
        pdf_path: Path to PDF file
        manual_id: Unique identifier for this manual
    """
    create_index()
    
    print(f"Processing PDF: {pdf_path}")
    pages = extract_text_by_page(pdf_path)
    print(f"Extracted {len(pages)} pages")
    
    actions = []
    chunk_count = 0
    
    for p in pages:
        # Extract metadata from page text
        section = infer_section(p["text"])
        chapter = infer_chapter(p["text"])
        part_number = infer_part_number(p["text"])
        
        # Create overlapping chunks
        chunks = chunk_text(p["text"], max_tokens=800, overlap=120)
        
        for chunk in chunks:
            # Generate 384-dim embedding
            vec = model.encode(chunk, normalize_embeddings=True).tolist()
            
            doc = {
                "_index": INDEX_NAME,
                "_id": str(uuid4()),
                "_source": {
                    "content": chunk,
                    "section": section,
                    "chapter": chapter,
                    "part_number": part_number,
                    "manual_id": manual_id,
                    "page": p["page"],
                    "embedding": vec
                }
            }
            actions.append(doc)
            chunk_count += 1
    
    # Bulk index all chunks
    helpers.bulk(es, actions)
    print(f"✓ Indexed {chunk_count} chunks from {len(pages)} pages")


def hybrid_search(query_text: str, k: int = 10) -> List[Dict]:
    """
    Execute hybrid search combining:
    - BM25 keyword search (match + match_phrase)
    - Vector similarity search (kNN)
    - Reciprocal Rank Fusion (RRF) for result merging
    
    Args:
        query_text: User query
        k: Number of results to return
    
    Returns:
        List of search results with content, page, section, part_number
    """
    # Generate query embedding
    qvec = model.encode(query_text, normalize_embeddings=True).tolist()
    
    # Hybrid search with RRF
    resp = es.search(
        index=INDEX_NAME,
        size=k,
        rank={
            "rrf": {
                "window_size": 100,
                "rank_constant": 60
            }
        },
        sub_searches=[
            {
                # BM25 keyword search
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"content": query_text}},
                            {"match_phrase": {"content": query_text}}
                        ],
                        "minimum_should_match": 1
                    }
                }
            },
            {
                # Vector similarity search
                "query": {
                    "knn": {
                        "field": "embedding",
                        "query_vector": qvec,
                        "k": 100,
                        "num_candidates": 1000
                    }
                }
            }
        ],
        _source=["content", "page", "section", "chapter", "manual_id", "part_number"]
    )
    
    return resp["hits"]["hits"]


if __name__ == "__main__":
    # Example usage
    print("=== Aviation Manual Ingestion Pipeline ===\n")
    
    # Index a PDF manual
    pdf_file = "sample_apu_manual.pdf"
    if os.path.exists(pdf_file):
        index_pdf(pdf_file, manual_id="APU_MANUAL_001")
    else:
        print(f"Note: {pdf_file} not found. Place your PDF in the same directory.")
    
    # Example hybrid search
    print("\n=== Testing Hybrid Search ===\n")
    query = "How do I reset the APU after a master warning?"
    results = hybrid_search(query, k=5)
    
    print(f"Query: {query}\n")
    print(f"Found {len(results)} results:\n")
    
    for i, r in enumerate(results, 1):
        src = r["_source"]
        score = r.get("_score", 0)
        print(f"{i}. [Page {src['page']}] Score: {score:.4f}")
        if src.get('section'):
            print(f"   Section: {src['section']}")
        if src.get('part_number'):
            print(f"   Part: {src['part_number']}")
        print(f"   Content: {src['content'][:200]}...")
        print()
