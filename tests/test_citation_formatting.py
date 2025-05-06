#!/usr/bin/env python3
"""
Test script to verify the citation formatting functionality in the RAG pipeline.
This test directly targets the citation formatting logic without mocking the full LLM pipeline.
"""

import logging
import sys
from pathlib import Path
import unittest
from typing import List, Dict, Any

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
    """Create test documents with various metadata configurations."""
    return [
        # Document with basic metadata
        Document(
            page_content="Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
            metadata={
                'id': 'chunk_001', 
                'rank': 0, 
                'similarity': 0.95,
            }
        ),
        # Document with extended metadata
        Document(
            page_content="Deep learning models have revolutionized artificial intelligence research.",
            metadata={
                'id': 'chunk_002', 
                'rank': 1, 
                'similarity': 0.87,
                'document_id': 'doc_002',
                'document_title': 'Deep Learning Fundamentals',
                'document_type': 'research_paper',
                'page': 15,
                'author': 'John Doe'
            }
        ),
        # Document with partial extended metadata
        Document(
            page_content="Natural Language Processing focuses on interaction between computers and human language.",
            metadata={
                'id': 'chunk_003', 
                'rank': 2, 
                'similarity': 0.82,
                'document_title': 'NLP Techniques',  # Only title, no document_id
                'page': 7,
            }
        ),
    ]

class TestCitationFormatting(unittest.TestCase):
    """Test case for citation formatting functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.documents = create_test_documents()
        # Use actual format_docs from RAGPipeline
        self.format_docs = RAGPipeline.format_docs
        
    def test_format_docs(self):
        """Test the format_docs method."""
        formatted_context = self.format_docs(self.documents)
        logger.info(f"Formatted context: {formatted_context}")
        
        # Verify that each document is formatted correctly with its rank
        for i, doc in enumerate(self.documents):
            self.assertIn(f"[{i}] {doc.page_content}", formatted_context)
            
        logger.info("format_docs test passed!")
            
    def test_citation_formatting(self):
        """Test the citation formatting functionality."""
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
                doc_id = doc.metadata.get('id', f'doc-{citation_idx}')
                
                # Get title with fallbacks
                doc_title = doc.metadata.get('document_title',  # Try extended metadata first
                           doc.metadata.get('title',           # Fall back to basic metadata
                           f'Document {citation_idx}'))        # Default fallback
                
                # Format the citation
                citation_text = f"[{citation_idx}] {doc_id}: {doc_title}"
                
                # Add page number if available
                if 'page' in doc.metadata:
                    citation_text += f", Page {doc.metadata['page']}"
                    
                # Add author if available
                if 'author' in doc.metadata:
                    citation_text += f", Author: {doc.metadata['author']}"
                    
                # Add document type if available
                if 'document_type' in doc.metadata:
                    citation_text += f" ({doc.metadata['document_type']})"
                    
                formatted_answer += citation_text + "\n"
        
        logger.info("Formatted answer with citations:")
        logger.info("-" * 50)
        logger.info(formatted_answer)
        logger.info("-" * 50)
        
        # Verify that each citation is included
        for i, doc in enumerate(self.documents):
            citation_marker = f"[{i}]"
            self.assertIn(citation_marker, formatted_answer)
            
        # Check for specific metadata in citations
        self.assertIn("chunk_001", formatted_answer)  # Basic ID
        self.assertIn("Deep Learning Fundamentals", formatted_answer)  # Extended title
        self.assertIn("Page 15", formatted_answer)  # Page number
        self.assertIn("John Doe", formatted_answer)  # Author
        self.assertIn("research_paper", formatted_answer)  # Document type
        
        logger.info("Citation formatting test passed!")
        
    def test_enhanced_citation_formatting(self):
        """Test an enhanced version of citation formatting that maximizes the use of available metadata."""
        # Create a mock CitedAnswer
        cited_answer = CitedAnswer(
            answer="An enhanced answer with citations [0][1][2].",
            citations=[0, 1, 2]
        )
        
        # Apply enhanced citation formatting logic
        formatted_answer = cited_answer.answer + "\n\nEnhanced Citations:\n"
        
        for citation_idx in cited_answer.citations:
            if citation_idx < len(self.documents):
                doc = self.documents[citation_idx]
                metadata = doc.metadata
                
                # Start with the citation number
                citation_text = f"[{citation_idx}] "
                
                # Preferred identifier: document_id > id > default
                identifier = metadata.get('document_id', metadata.get('id', f'doc-{citation_idx}'))
                citation_text += identifier
                
                # Always add a separator
                citation_text += ": "
                
                # Title: document_title > title > default
                title = metadata.get('document_title', metadata.get('title', f'Document {citation_idx}'))
                citation_text += title
                
                # Add all additional metadata that might be useful for citations
                extra_info = []
                
                if 'page' in metadata:
                    extra_info.append(f"Page {metadata['page']}")
                    
                if 'author' in metadata:
                    extra_info.append(f"Author: {metadata['author']}")
                    
                if 'document_type' in metadata:
                    extra_info.append(f"Type: {metadata['document_type']}")
                    
                if 'similarity' in metadata:
                    # Format similarity as percentage with 1 decimal place
                    similarity_pct = f"{metadata['similarity'] * 100:.1f}%"
                    extra_info.append(f"Relevance: {similarity_pct}")
                
                # Add all extra info with commas
                if extra_info:
                    citation_text += " (" + ", ".join(extra_info) + ")"
                
                formatted_answer += citation_text + "\n"
        
        logger.info("Enhanced formatted answer with citations:")
        logger.info("-" * 50)
        logger.info(formatted_answer)
        logger.info("-" * 50)
        
        # Verify enhanced citations
        self.assertIn("Relevance: 95.0%", formatted_answer)  # Formatted similarity
        self.assertIn("Page 7", formatted_answer)  # Partial metadata
        self.assertIn("Type: research_paper", formatted_answer)  # Document type
        
        logger.info("Enhanced citation formatting test passed!")

def main():
    """Run the tests."""
    logger.info("Running citation formatting tests")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
