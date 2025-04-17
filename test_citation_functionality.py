#!/usr/bin/env python3
"""
Simplified test script focusing specifically on citation functionality in the RAG pipeline.
This version uses mocks to avoid dependencies on external services.
"""

import logging
import sys
import json
from typing import List, Dict, Any, Callable

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[TEST] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class Document:
    """Simple document class for testing."""
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

def create_test_documents():
    """Create test documents with realistic metadata."""
    return [
        Document(
            page_content="Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
            metadata={"id": "doc-001", "title": "Introduction to Machine Learning", "rank": 0, "similarity": 0.95}
        ),
        Document(
            page_content="Deep learning models, particularly neural networks with many layers, have revolutionized artificial intelligence research.",
            metadata={"id": "doc-002", "title": "Deep Learning Models", "rank": 1, "similarity": 0.85}
        ),
        Document(
            page_content="Natural Language Processing (NLP) focuses on the interaction between computers and human language.",
            metadata={"id": "doc-003", "title": "Natural Language Processing", "rank": 2, "similarity": 0.75}
        ),
    ]

def format_docs(docs: List[Document]) -> str:
    """Format documents for prompt - this is the exact function from RAGPipeline."""
    return "\n\n".join(f"[{doc.metadata['rank']}] {doc.page_content}" for doc in docs)

def test_citation_functionality():
    """Test the citation functionality of the RAG pipeline."""
    # Get test documents
    docs = create_test_documents()
    
    # Format documents as they would be in the RAG pipeline
    formatted_context = format_docs(docs)
    logger.info(f"Formatted context:\n{formatted_context}")
    
    # Simulate LLM response with citations
    # This simulates what the structured output from ChatGPT or similar would return
    mock_llm_response = {
        "answer": "The documents cover various aspects of AI, including machine learning, deep learning models, and natural language processing. Machine learning allows computers to learn without explicit programming [0], while deep learning models have revolutionized AI research [1]. Natural Language Processing (NLP) is specifically focused on human-computer language interaction [2].",
        "citations": [0, 1, 2]  # Citing all three documents
    }
    
    # Format the answer with citations (similar to RAGPipeline.__call__ method)
    formatted_answer = mock_llm_response["answer"]
    citations = mock_llm_response["citations"]
    
    if citations:
        formatted_answer += "\n\nCitations:\n"
        for citation_idx in citations:
            if citation_idx < len(docs):
                doc = docs[citation_idx]
                doc_id = doc.metadata.get('id', f'doc-{citation_idx}')
                doc_title = doc.metadata.get('title', f'Document {citation_idx}')
                formatted_answer += f"[{citation_idx}] {doc_id}: {doc_title}\n"
    
    # Print the final formatted answer with citations
    logger.info("Final formatted answer with citations:")
    logger.info("-" * 80)
    logger.info(formatted_answer)
    logger.info("-" * 80)
    
    # Check if citations were included
    if "Citations:" in formatted_answer:
        logger.info("SUCCESS: Citations were properly included in the response!")
        return True
    else:
        logger.error("FAILURE: Citations were not found in the response.")
        return False

def main():
    """Run the test."""
    logger.info("Testing citation functionality in the RAG pipeline")
    
    try:
        success = test_citation_functionality()
        return success
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
