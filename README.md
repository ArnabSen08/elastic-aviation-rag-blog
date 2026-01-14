# Beyond Keywords: Building an AI Assistant for Aviation Maintenance using Elastic RAG

**Elastic Blog-a-thon Contest Entry**

This repository contains the complete implementation of an AI-powered aviation maintenance assistant using Elasticsearch's hybrid search capabilities (BM25 + Vector Search + RRF).

## 📋 Contents

1. **Elasticsearch Query DSL** - Hybrid search with RRF
2. **Python Ingestion Pipeline** - PDF parsing, chunking, and indexing
3. **Architecture Diagram** - System flow description

## 🚀 Quick Links

- [Elasticsearch Hybrid Query](./elasticsearch_hybrid_query.json)
- [Python Ingestion Code](./ingest_aviation_manuals.py)
- [Architecture Description](./ARCHITECTURE.md)

## 📊 Key Features

- **Hybrid Search**: Combines BM25 keyword matching with 384-dim vector embeddings
- **Reciprocal Rank Fusion (RRF)**: Merges multiple ranking signals
- **Metadata Preservation**: Extracts page numbers, sections, ATA chapters, part numbers
- **Overlapping Chunks**: 800-word chunks with 120-word overlap for context continuity

## 🛠️ Tech Stack

- Elasticsearch 8.x
- Python 3.8+
- sentence-transformers/all-MiniLM-L6-v2
- PyPDF2

## 📖 Blog Post

Read the full technical blog post: [Link to be added]

---

**Author**: [Your Name]  
**Contest**: Elastic Blog-a-thon 2026
