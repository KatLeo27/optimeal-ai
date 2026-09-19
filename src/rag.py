"""
OptiMeal AI - Sustainability RAG Assistant Engine
Performs document indexing, semantic vector/TF-IDF retrieval, and cited query answering
over curated sustainability, EPA, and food recovery guidelines.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import logging
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DOCS_DIR = os.path.join("knowledge_base", "documents")

class SustainabilityRAGAssistant:
    """
    Retrieval-Augmented Generation assistant for sustainable food service guidelines.
    """
    def __init__(self, docs_dir: str = DOCS_DIR):
        self.docs_dir = docs_dir
        self.chunks = []
        self.metadata = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = None
        self._index_documents()

    def _index_documents(self):
        """
        Load text documents from knowledge base and partition into retrievable chunks.
        """
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)
            logger.warning(f"Knowledge base directory {self.docs_dir} was empty.")
            return

        doc_files = glob.glob(os.path.join(self.docs_dir, "*.txt")) + glob.glob(os.path.join(self.docs_dir, "*.md"))
        
        for filepath in doc_files:
            filename = os.path.basename(filepath)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Split by sections / double newlines into semantic passages
                sections = [s.strip() for s in content.split("\n\n") if len(s.strip()) > 40]
                
                for idx, sec in enumerate(sections):
                    self.chunks.append(sec)
                    self.metadata.append({
                        "source_file": filename,
                        "chunk_id": idx + 1,
                        "preview": sec[:120] + "..." if len(sec) > 120 else sec
                    })
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")
                
        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
            logger.info(f"Indexed {len(self.chunks)} knowledge chunks from {len(doc_files)} files.")
        else:
            logger.warning("No document chunks available for indexing.")

    def query_knowledge_base(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve relevant knowledge base chunks and format response with sources.
        """
        if not self.chunks or self.tfidf_matrix is None:
            return {
                "answer": "No documents are currently indexed in the knowledge base.",
                "retrieved_sources": [],
                "confidence": 0.0
            }

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Get top-k indices sorted descending
        top_indices = similarities.argsort()[::-1][:top_k]
        
        retrieved_sources = []
        top_score = float(similarities[top_indices[0]]) if len(top_indices) > 0 else 0.0
        
        if top_score < 0.05:
            return {
                "query": query,
                "answer": (
                    "I could not find sufficiently relevant guidelines in the local sustainability knowledge base for your query. "
                    "Try asking about food recovery hierarchy, batch cooking, plate waste prevention, donation safety, or SDG 12 targets."
                ),
                "retrieved_sources": [],
                "confidence": round(top_score, 3)
            }

        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.03:
                retrieved_sources.append({
                    "source_file": self.metadata[idx]["source_file"],
                    "chunk_id": self.metadata[idx]["chunk_id"],
                    "relevance_score": round(score * 100, 1),
                    "content": self.chunks[idx]
                })

        # Synthesize clear answer text combining top matches
        primary_chunk = retrieved_sources[0]["content"] if retrieved_sources else ""
        source_names = list(set(s["source_file"] for s in retrieved_sources))
        
        answer_text = (
            f"Based on **{', '.join(source_names)}**:\n\n"
            f"{primary_chunk}"
        )
        
        if len(retrieved_sources) > 1:
            answer_text += f"\n\n**Additional Guideline:**\n{retrieved_sources[1]['content']}"

        return {
            "query": query,
            "answer": answer_text,
            "retrieved_sources": retrieved_sources,
            "confidence": round(top_score, 3)
        }

if __name__ == "__main__":
    assistant = SustainabilityRAGAssistant()
    sample_q = "What are common strategies to reduce food waste in cafeterias?"
    res = assistant.query_knowledge_base(sample_q)
    print(f"Query: {sample_q}")
    print(f"\nAnswer:\n{res['answer']}")
    print(f"\nSources ({len(res['retrieved_sources'])}):")
    for s in res['retrieved_sources']:
        print(f" - {s['source_file']} (Score: {s['relevance_score']}%)")
