"""
Aviation Manual Processing Pipeline for Elasticsearch Hybrid Search
Blog-ready code demonstrating PDF parsing, chunking, embedding, and indexing
"""

import re
from typing import List, Dict
from uuid import uuid4

import PyPDF2
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

class AviationManualProcessor:
    def __init__(self, es_host="http://localhost:9200", index_name="aviation_manuals"):
        self.es = Elasticsearch(es_host)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index_name = index_name
        self._create_index()
    
    def _create_index(self):
        """Create Elasticsearch index with hybrid search mappings"""
        if self.es.indices.exists(index=self.index_name):
            return
            
        mapping = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "section": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "part_number": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "embedding": {"type": "dense_vector", "dims": 384, "similarity": "cosine"}
                }
            }
        }
        self.es.indices.create(index=self.index_name, body=mapping)
        print(f"✓ Created index: {self.index_name}")
    
    def parse_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF by page"""
        pages = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    # Clean whitespace
                    text = re.sub(r"\s+", " ", text).strip()
                    pages.append({"page": i, "text": text})
        return pages
    
    def chunk_text(self, text: str, max_words=800, overlap=120) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
            start = end - overlap
        
        return [c for c in chunks if len(c.split()) > 50]
    
    def extract_metadata(self, text: str) -> Dict:
        """Extract aviation-specific metadata from text"""
        # Section patterns: "SECTION 3.2: Engine Systems"
        section_match = re.search(r"SECTION\s+\d+[\.\d]*\s*[:\-]\s*[A-Z][A-Za-z0-9\-\s]+", text, re.IGNORECASE)
        section = section_match.group(0).strip() if section_match else ""
        
        # ATA Chapter patterns: "ATA Chapter 49" or "ATA 49"
        ata_match = re.search(r"ATA\s*(?:Chapter\s*)?\d{2}", text, re.IGNORECASE)
        ata_chapter = ata_match.group(0).strip() if ata_match else ""
        
        # Part number patterns: "APU-MSTR-RESET"
        part_match = re.search(r"\b([A-Z]{2,}-[A-Z0-9]{2,}[A-Z0-9\-]*)\b", text)
        part_number = part_match.group(1) if part_match else ""
        
        return {
            "section": section,
            "ata_chapter": ata_chapter,
            "part_number": part_number
        }
    
    def process_manual(self, pdf_path: str, manual_id: str):
        """Complete processing pipeline: parse → chunk → embed → index"""
        print(f"Processing: {pdf_path}")
        
        # 1. Parse PDF
        pages = self.parse_pdf(pdf_path)
        print(f"✓ Extracted {len(pages)} pages")
        
        # 2. Process each page
        documents = []
        for page_data in pages:
            # Extract metadata
            metadata = self.extract_metadata(page_data["text"])
            
            # 3. Create overlapping chunks
            chunks = self.chunk_text(page_data["text"])
            
            for chunk in chunks:
                # 4. Generate embeddings (384-dimensional)
                embedding = self.model.encode(chunk, normalize_embeddings=True).tolist()
                
                doc = {
                    "_index": self.index_name,
                    "_id": str(uuid4()),
                    "_source": {
                        "content": chunk,
                        "page": page_data["page"],
                        "section": metadata["section"],
                        "ata_chapter": metadata["ata_chapter"],
                        "part_number": metadata["part_number"],
                        "manual_id": manual_id,
                        "embedding": embedding
                    }
                }
                documents.append(doc)
        
        # 5. Bulk index to Elasticsearch
        helpers.bulk(self.es, documents)
        print(f"✓ Indexed {len(documents)} chunks")
    
    def hybrid_search(self, query: str, k=10) -> List[Dict]:
        """Execute hybrid search with BM25 + vector search + RRF"""
        # Generate query embedding
        query_vector = self.model.encode(query, normalize_embeddings=True).tolist()
        
        # Hybrid search query
        response = self.es.search(
            index=self.index_name,
            size=k,
            rank={
                "rrf": {"window_size": 100, "rank_constant": 60}
            },
            sub_searches=[
                {
                    # BM25 keyword search
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {"content": query}},
                                {"match_phrase": {"content": query}}
                            ]
                        }
                    }
                },
                {
                    # Vector similarity search
                    "query": {
                        "knn": {
                            "field": "embedding",
                            "query_vector": query_vector,
                            "k": 100,
                            "num_candidates": 1000
                        }
                    }
                }
            ],
            _source=["content", "page", "section", "part_number"]
        )
        
        return response["hits"]["hits"]


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = AviationManualProcessor()
    
    # Process a manual
    processor.process_manual("apu_manual.pdf", "APU_001")
    
    # Search examples
    queries = [
        "How to reset APU master warning?",
        "Hydraulic system pressure loss troubleshooting",
        "Engine start procedure checklist"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = processor.hybrid_search(query, k=3)
        
        for i, hit in enumerate(results, 1):
            source = hit["_source"]
            print(f"{i}. Page {source['page']} | Score: {hit['_score']:.3f}")
            if source["section"]:
                print(f"   Section: {source['section']}")
            if source["part_number"]:
                print(f"   Part: {source['part_number']}")
            print(f"   Content: {source['content'][:150]}...")