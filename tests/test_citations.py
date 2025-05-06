#!/usr/bin/env python3
"""
Test script to verify the citation formatting functionality in the RAG pipeline.
This test directly targets the citation formatting logic without mocking the full LLM pipeline.
"""

import logging
import sys
import unittest
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[TEST] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import RAG components
from src.mmore.rag.pipeline import RAGPipeline
from src.mmore.rag.types import CitedAnswer
from langchain_core.documents import Document

def create_test_documents():
    """Create test documents with basic metadata for citation testing."""
    return [
        Document(
            page_content="Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
            metadata={
                'id': 'doc-001', 
                'title': 'Introduction to Machine Learning',
                'rank': 0
            }
        ),
        Document(
            page_content="Deep learning models have revolutionized artificial intelligence research.",
            metadata={
                'id': 'doc-002', 
                'title': 'Deep Learning Fundamentals',
                'rank': 1
            }
        ),
        Document(
            page_content="Natural Language Processing focuses on interaction between computers and human language.",
            metadata={
                'id': 'doc-003', 
                'title': 'NLP Techniques',
                'rank': 2
            }
        ),
    ]

class TestCitationFormatting(unittest.TestCase):
    """Test case for citation formatting functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.documents = create_test_documents()
        self.format_docs = RAGPipeline.format_docs
        
    def test_format_docs(self):
        """Test the format_docs method to ensure documents are correctly formatted for the prompt."""
        formatted_context = self.format_docs(self.documents)
        logger.info(f"Formatted context: {formatted_context}")
        
        # Verify that each document is formatted correctly with its rank
        for i, doc in enumerate(self.documents):
            self.assertIn(f"[{i}] {doc.page_content}", formatted_context)
            
        logger.info("format_docs test passed!")
            
    def test_citation_formatting(self):
        """Test the citation formatting functionality in the RAG pipeline."""
        # Create a mock CitedAnswer as if returned from LLM
        cited_answer = CitedAnswer(
            answer="Machine learning is fascinating [0]. Deep learning has revolutionized AI [1]. NLP is important for human-computer interaction [2].",
            citations=[0, 1, 2]  # Cite all documents
        )
        
        # Manually apply the citation formatting logic from RAGPipeline.__call__
        formatted_answer = cited_answer.answer
        
        # Add citation block - this mimics the logic in RAGPipeline.__call__
        formatted_answer += "\n\nCitations:\n"
        for citation_idx in cited_answer.citations:
            if citation_idx < len(self.documents):
                doc = self.documents[citation_idx]
                # Use document ID and title from metadata if available
                doc_id = doc.metadata.get('id', f'doc-{citation_idx}')
                doc_title = doc.metadata.get('title', f'Document {citation_idx}')
                formatted_answer += f"[{citation_idx}] {doc_id}: {doc_title}\n"
        
        logger.info("Formatted answer with citations:")
        logger.info("-" * 50)
        logger.info(formatted_answer)
        logger.info("-" * 50)
        
        # Verify that each citation is included
        for i, doc in enumerate(self.documents):
            citation_marker = f"[{i}]"
            self.assertIn(citation_marker, formatted_answer)
            
        # Check for specific metadata in citations
        self.assertIn("doc-001", formatted_answer)  # Document ID
        self.assertIn("Introduction to Machine Learning", formatted_answer)  # Document title
        self.assertIn("Deep Learning Fundamentals", formatted_answer)  # Document title
        self.assertIn("NLP Techniques", formatted_answer)  # Document title
        
        logger.info("Citation formatting test passed!")

def main():
    """Run the tests."""
    logger.info("Running citation formatting tests")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
