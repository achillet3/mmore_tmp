#!/usr/bin/env python3
"""
Simplified test script for RAG citation functionality.
This avoids complex dependencies and focuses just on the citation mechanism.
"""

import logging
import sys
import os
from typing import List, Dict, Any

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

# Import OpenAI directly to avoid LangChain dependency issues
try:
    from openai import OpenAI
except ImportError:
    logger.error("OpenAI Python package not found. Please install it with: pip install openai")
    sys.exit(1)

# Define document class
class SimpleDocument:
    """Simple document class with content and metadata."""
    
    def __init__(self, content, doc_id, title, page=None, author=None):
        self.content = content
        self.metadata = {
            "id": doc_id,
            "title": title
        }
        if page is not None:
            self.metadata["page"] = page
        if author is not None:
            self.metadata["author"] = author

# Test documents
test_documents = [
    SimpleDocument(
        "Machine learning is a subset of artificial intelligence focused on building systems that learn from data.",
        "doc-1", 
        "Introduction to Machine Learning",
        page=1,
        author="Smith, J."
    ),
    SimpleDocument(
        "Deep learning uses neural networks with many layers to analyze various factors of data.",
        "doc-2", 
        "Deep Learning Fundamentals",
        page=15,
        author="Chen, L."
    ),
    SimpleDocument(
        "Natural language processing (NLP) is a field of AI that enables computers to understand human language.",
        "doc-3", 
        "NLP Basics",
        page=42,
        author="Johnson, M."
    )
]

def create_context(documents):
    """Create a context string from documents."""
    context_parts = []
    for i, doc in enumerate(documents):
        context_parts.append(f"[{i}] {doc.content}")
    return "\n\n".join(context_parts)

def create_citation_prompt(enforce_citations=False):
    """Create a prompt that includes citation instructions."""
    if enforce_citations:
        system_prompt = """You are a helpful AI assistant that MUST cite your sources. 
For EVERY piece of information you provide, you MUST cite at least one document from the context using its index.
Even if you think you know the answer, you MUST cite the provided context documents.
YOU MUST RESPOND IN THIS EXACT JSON FORMAT: 
{"answer": "your detailed answer with explicit citations like [0], [1], etc.", "citations": [0, 1, 2]}
The answer should include citations in square brackets [0], [1], etc. directly in the text.
The citations array must contain all the document indices you referenced.
If no documents in the context are relevant, cite document [0] anyway.
THIS FORMAT IS CRUCIAL. DO NOT deviate from it or add any other text."""
    else:
        system_prompt = """You are a helpful AI assistant that provides accurate information based on the context provided.
When you reference information from the context, please include a citation.
Respond using JSON format with an 'answer' field and a 'citations' array listing the indices of documents you referenced."""
    
    return system_prompt

def format_citations(answer, citations, documents):
    """Format the answer with proper citations."""
    formatted_answer = answer
    
    # Add citation block if citations exist
    if citations:
        formatted_answer += "\n\nCitations:\n"
        for citation_idx in citations:
            if citation_idx < len(documents):
                doc = documents[citation_idx]
                doc_id = doc.metadata.get("id", f"doc-{citation_idx}")
                doc_title = doc.metadata.get("title", f"Document {citation_idx}")
                
                formatted_answer += f"[{citation_idx}] {doc_id}: {doc_title}\n"
                
                # Add page and author if available
                if "page" in doc.metadata:
                    formatted_answer += f"    Page: {doc.metadata['page']}\n"
                if "author" in doc.metadata:
                    formatted_answer += f"    Author: {doc.metadata['author']}\n"
    
    return formatted_answer

def get_answer_with_citations(query, documents, enforce_citations=False):
    """Get an answer with citations for a query using the given documents."""
    import json
    import re
    
    # Create client
    client = OpenAI()
    
    # Create context from documents
    context = create_context(documents)
    
    # Create system prompt
    system_prompt = create_citation_prompt(enforce_citations)
    
    # Create messages for chat completion
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {query}\n\nContext: {context}"}
    ]
    
    # Get completion
    logger.info(f"Sending query to OpenAI: {query}")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    
    # Extract content
    content = response.choices[0].message.content
    logger.info(f"Raw response: {content}")
    
    # Try to parse as JSON
    try:
        # Look for JSON pattern in the text
        json_match = re.search(r'\{[^\{\}]*\}', content)
        if json_match:
            json_str = json_match.group(0)
            logger.info(f"Found potential JSON: {json_str}")
            json_data = json.loads(json_str)
            answer = json_data.get('answer', content)
            citations = json_data.get('citations', [])
            logger.info(f"Extracted citations: {citations}")
        else:
            # Try to extract citation markers if JSON parsing fails
            answer = content
            citation_markers = re.findall(r'\[(\d+)\]', content)
            citations = list(set([int(marker) for marker in citation_markers]))
            if not citations:
                logger.warning("No citations found in response")
    except Exception as e:
        logger.error(f"Error parsing response as JSON: {e}")
        answer = content
        citations = []
    
    # Format the answer with citations
    formatted_answer = format_citations(answer, citations, documents)
    
    return formatted_answer, bool(citations)

def main():
    """Run citation tests with both enforcement settings."""
    # Test query
    query = "What is machine learning and how does it relate to deep learning?"
    
    # Test with citation enforcement on
    logger.info("\n===== Testing with citation enforcement ON =====")
    formatted_answer1, has_citations1 = get_answer_with_citations(
        query, test_documents, enforce_citations=True
    )
    logger.info(f"\nFormatted answer:\n{formatted_answer1}")
    if has_citations1:
        logger.info("SUCCESS: Response includes citations")
    else:
        logger.error("ERROR: Response does not include citations")
    
    # Test with citation enforcement off
    logger.info("\n===== Testing with citation enforcement OFF =====")
    formatted_answer2, has_citations2 = get_answer_with_citations(
        query, test_documents, enforce_citations=False
    )
    logger.info(f"\nFormatted answer:\n{formatted_answer2}")
    if has_citations2:
        logger.info("SUCCESS: Response includes citations")
    else:
        logger.error("ERROR: Response does not include citations")
    
    # Report results
    if has_citations1 and not has_citations2:
        logger.info("\n✓ Citations only enforced when enabled - WORKING AS EXPECTED")
    elif has_citations1 and has_citations2:
        logger.info("\nCitations provided in both cases - Regular prompting is sufficient")
    elif not has_citations1 and not has_citations2:
        logger.error("\n❌ Citations not provided in either case - Something is wrong")
    else:  # not has_citations1 and has_citations2
        logger.error("\n❌ Unexpected behavior: Citations only provided when NOT enforced")

if __name__ == "__main__":
    main()
