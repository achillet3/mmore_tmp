#!/usr/bin/env python3
"""
Simple test script to verify that the RAG citation pipeline works with the updated prompt.
This test creates a mock retriever with pre-defined documents and tests that citations are generated.
"""

import logging
import sys
from typing import Dict, List, Any, Optional
import os

# Handle various LangChain import paths based on installed version
try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
except ImportError:
    # Fall back to older import paths
    try:
        from langchain.schema import Document
        from langchain.schema.retriever import BaseRetriever
        from langchain.callbacks.manager import CallbackManagerForRetrieverRun
    except ImportError:
        logging.error("Could not import required LangChain classes. Check your installation.")
        sys.exit(1)

# Try to import ChatOpenAI from different locations
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        logging.error("Could not import ChatOpenAI. Check your installation.")
        sys.exit(1)

from src.mmore.rag.pipeline import RAGPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[CITATION_TEST] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Check for OpenAI API key
if not os.getenv('OPENAI_API_KEY'):
    logger.warning("OPENAI_API_KEY not found in environment. Please set it before running this test.")
    api_key = input("Enter your OpenAI API key: ").strip()
    if not api_key:
        logger.error("No API key provided. Exiting.")
        sys.exit(1)
    os.environ['OPENAI_API_KEY'] = api_key
    logger.info("API key set for this session.")
else:
    logger.info("OpenAI API key found in environment.")

# Create a simple test retriever with predefined documents
class MockRetriever(BaseRetriever):
    """A simple mock retriever that returns predefined documents."""
    
    # Set some class variables for Pydantic compatibility
    k: int = 3
    _test_docs: List[Document] = None
    
    def __init__(self):
        """Initialize with predefined documents."""
        super().__init__()
        self._test_docs = [
            Document(
                page_content="Machine learning is a subset of artificial intelligence focused on building systems that learn from data.",
                metadata={"id": "doc-1", "title": "Introduction to Machine Learning"}
            ),
            Document(
                page_content="Deep learning uses neural networks with many layers to analyze various factors of data.",
                metadata={"id": "doc-2", "title": "Deep Learning Fundamentals"}
            ),
            Document(
                page_content="Natural language processing (NLP) is a field of AI that enables computers to understand human language.",
                metadata={"id": "doc-3", "title": "NLP Basics"}
            )
        ]
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Return all predefined documents regardless of the query."""
        logger.info(f"Mock retriever returning {len(self._test_docs)} documents for query: {query}")
        return self._test_docs


def test_rag_pipeline(enforce_citations=True):
    """Test the RAG pipeline with the updated prompt template."""
    logger.info(f"Testing RAG pipeline with enforce_citations={enforce_citations}")
    
    # Create components
    retriever = MockRetriever()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create the RAG pipeline
    rag_pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        enforce_citations=enforce_citations
    )
    
    # Test query
    query = "What is machine learning and how does it relate to deep learning?"
    logger.info(f"Running test query: {query}")
    
    # Get result
    result = rag_pipeline({"input": query})
    
    if isinstance(result, list):
        result = result[0]  # Take first result if multiple
    
    logger.info(f"Result: {result}")
    
    # Check for citation formatting
    if "Citation" in result or "[" in result and "]" in result:
        logger.info("SUCCESS: Response includes citations")
        return True
    else:
        logger.error("ERROR: Response does not include citations")
        return False

if __name__ == "__main__":
    # Test with citation enforcement on
    success_with = test_rag_pipeline(enforce_citations=True)
    
    # Test with citation enforcement off
    success_without = test_rag_pipeline(enforce_citations=False)
    
    # Report results
    if success_with and not success_without:
        logger.info("✓ Citations only enforced when enabled - WORKING AS EXPECTED")
    elif success_with and success_without:
        logger.info("Citations provided in both cases - Regular prompting is sufficient")
    elif not success_with and not success_without:
        logger.error("❌ Citations not provided in either case - Something is wrong")
    else:  # not success_with and success_without
        logger.error("❌ Unexpected behavior: Citations only provided when NOT enforced")
        
    sys.exit(0 if success_with else 1)
