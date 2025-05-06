#!/usr/bin/env python3
"""
Test script to run the RAG pipeline with citations using real Milvus data.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import functools

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[RAG_TEST] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import necessary components
try:
    from pymilvus import MilvusClient
    from src.mmore.index.indexer import get_model_from_index, DBConfig
    from src.mmore.rag.model import DenseModelConfig, SparseModelConfig
    from src.mmore.rag.retriever import Retriever, RetrieverConfig
    from src.mmore.rag.llm import LLM, LLMConfig
    from src.mmore.rag.pipeline import RAGPipeline
    from src.mmore.rag.types import CitedAnswer
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
except ImportError as e:
    logger.error(f"Error importing required packages: {e}")
    logger.error("Make sure you're using the correct Python environment (mmore-py310).")
    sys.exit(1)

def check_openai_api_key():
    """Check if OpenAI API key is set in environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY environment variable not set.")
        api_key = input("Please enter your OpenAI API key: ").strip()
        if not api_key:
            logger.error("No API key provided. Exiting.")
            sys.exit(1)
        os.environ["OPENAI_API_KEY"] = api_key
        logger.info("API key has been set for this session.")
    else:
        logger.info("OpenAI API key found in environment variables.")

# Create a simpler test retriever that doesn't use Pydantic validation
class SimpleTestRetriever(BaseRetriever):
    """Simple retriever that uses direct Milvus search with dense embeddings only."""

    def __init__(self, client, dense_model, collection_name="test_collection", k=3, output_fields=None):
        # Initialize as parent class first
        super().__init__()
        # Then set our own attributes
        self._client = client
        self._dense_model = dense_model
        self._collection_name = collection_name
        self._k = k
        # Match the field names used in the Milvus collection
        self._output_fields = output_fields or ["doc_id", "title", "document_type", "page", "author"]
    
    def _get_relevant_documents(
        self, query: Dict[str, Any],
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents from Milvus using dense search."""
        if self._k == 0:
            return []
        
        # Get the query text - handle both direct string queries and dict with 'input' key
        query_text = query.get("input", query) if isinstance(query, dict) else query
        logger.info(f"Retrieving documents for query: {query_text}")

        try:
            # Encode the query text using the dense model
            query_embedding = self._dense_model.encode(query_text) if hasattr(self._dense_model, 'encode') else self._dense_model.embed_query(query_text)
            
            # Check if we need to convert to list (some models already return lists)
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()

            # Perform Milvus search - using search_params instead of param to avoid conflict
            # Use COSINE metric type as expected by the index
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self._client.search(
                collection_name=self._collection_name,
                data=[query_embedding],  # Already converted to list if needed
                anns_field="dense_embedding",
                search_params=search_params,  # Use search_params not param
                limit=self._k,
                output_fields=self._output_fields + ["text"]  # Make sure to include the text field
            )
            logger.info(f"Retrieved {len(results[0])} documents")

            # Convert Milvus results to Documents
            docs = []
            for i, hit in enumerate(results[0]):
                try:
                    # Milvus returns a dict for each hit, not an entity object
                    # Create metadata from all output fields
                    metadata = {}
                    for field in self._output_fields:
                        if field in hit:
                            metadata[field] = hit[field]
                    
                    # Add rank and similarity score
                    metadata["rank"] = i + 1
                    metadata["similarity"] = hit["distance"]
                    metadata["id"] = hit.get("doc_id", f"doc-{i}")  # Ensure id is set for citation

                    # Get the document text
                    text = hit.get("text", f"Document content {i}")
                    
                    docs.append(Document(
                        page_content=text,
                        metadata=metadata
                    ))
                except Exception as e:
                    logger.error(f"Error processing Milvus result: {e}")
                    logger.error(f"Result that caused error: {hit}")

            return docs
        except Exception as e:
            logger.error(f"Error during Milvus search: {e}")
            # Return an empty list if there's an error
            return []

def create_rag_pipeline():
    """Create the RAG pipeline using the real Milvus database."""
    logger.info("Creating RAG pipeline with real Milvus retriever")
    
    # 1. Configure database connection
    db_config = DBConfig(
        uri="http://localhost:19530",
        name="default"  # Use default database
    )
    
    # 2. Create Milvus client
    client = MilvusClient(uri=db_config.uri, db_name=db_config.name)
    logger.info("Connected to Milvus server")
    
    # 3. Set up collection name
    collection_name = "test_collection"
    logger.info(f"Using collection: {collection_name}")
    
    # 4. Get model info from custom metadata fields in collection
    try:
        # Get collection info to retrieve model config fields - don't filter by ID
        docs = client.query(
            collection_name=collection_name,
            filter="",  # Get any document since they all have the same model info
            output_fields=["dense_model_name", "sparse_model_name", "is_multimodal"],
            limit=1
        )
        
        if not docs:
            raise ValueError(f"No documents found in collection {collection_name}")
        
        # Get model config from document fields
        doc = docs[0]
        
        # 5. Create model configs
        dense_model_config = DenseModelConfig(
            model_name=doc["dense_model_name"],
            is_multimodal=(doc["is_multimodal"] == "True"),
        )
        
        sparse_model_config = SparseModelConfig(
            model_name=doc["sparse_model_name"],
            is_multimodal=(doc["is_multimodal"] == "True"),
        )
        
        # 6. Create the dense model
        from src.mmore.rag.model import DenseModel
        dense_model = DenseModel.from_config(dense_model_config)
        logger.info(f"Created dense model: {dense_model_config.model_name}")
        
        # 7. Create our custom retriever that directly uses Milvus search
        logger.info(f"Creating custom SimpleTestRetriever for collection: {collection_name}")
        retriever = SimpleTestRetriever(
            client=client,
            dense_model=dense_model,
            collection_name=collection_name,
            k=3,
            output_fields=["doc_id", "title", "document_type", "page", "author"]
        )
        
        logger.info("Successfully created retriever")
        
        # 8. Create and test the RAG pipeline
        logger.info("Creating RAG pipeline")
        
        # Import the DEFAULT_PROMPT from pipeline.py to use the simplified citation-friendly prompt
        from src.mmore.rag.pipeline import DEFAULT_PROMPT
        
        # Create a prompt template using the DEFAULT_PROMPT
        # Let the structured output with CitedAnswer model handle the format
        prompt = ChatPromptTemplate.from_messages([
            ("system", DEFAULT_PROMPT),
            ("human", "Question: {input}\n\nContext: {context}")
        ])
        
        logger.info("Using default prompt with citation instructions, output will be structured with CitedAnswer model")
        
        # 7. Configure LLM
        llm_config = LLMConfig(
            llm_name="gpt-3.5-turbo",
            temperature=0.0,  # For consistent results during testing
            max_new_tokens=500
        )
        
        # 8. Create LLM using the from_config class method
        llm = LLM.from_config(llm_config)
        
        # 9. Create RAG pipeline
        logger.info("Creating RAG pipeline")
        rag_pipeline = RAGPipeline(
            retriever=retriever,
            prompt_template=prompt,
            llm=llm
        )
        
        return rag_pipeline, retriever, client
        
    finally:
        # Nothing to clean up in the finally block since we're not patching anything
        pass

def run_test_queries(rag_pipeline, queries, client=None):
    """Run test queries through the RAG pipeline and check citations."""
    import traceback
    import json
    import re
    
    logger.info(f"Running {len(queries)} test queries")
    
    citation_success = 0
    total_queries = len(queries)
    results = []
    
    # Create custom wrapper to handle citation format issues
    def process_query(query_str):
        try:
            # Get the documents from retriever
            docs = rag_pipeline.retriever._get_relevant_documents({"input": query_str}, run_manager=None)
            
            # Format the context
            context = rag_pipeline.format_docs(docs)
            
            # Format the prompt
            prompt_content = rag_pipeline.prompt.invoke({"input": query_str, "context": context})
            
            # Get raw answer from LLM
            response = rag_pipeline.llm.invoke(prompt_content).content
            logger.info(f"Raw LLM response: {response}")
            
            # Try to parse JSON format from the response
            try:
                # Look for a JSON object in the response
                json_match = re.search(r'\{[^\{\}]*\}', response)
                if json_match:
                    json_str = json_match.group(0)
                    logger.info(f"Found potential JSON: {json_str}")
                    json_data = json.loads(json_str)
                    answer = json_data.get('answer', response)
                    citations = json_data.get('citations', [])
                    logger.info(f"Extracted citations: {citations}")
                else:
                    # Try to extract citation markers if JSON parsing fails
                    answer = response
                    citation_markers = re.findall(r'\[(\d+)\]', answer)
                    citations = [int(marker) for marker in citation_markers]
                    if not citations:
                        logger.warning("No citations found in response")
            except Exception as e:
                logger.error(f"Error parsing response as JSON: {e}")
                answer = response
                citations = []
            
            # Format the answer with citations
            formatted_answer = answer
            
            if citations:
                formatted_answer += "\n\nCitations:\n"
                for idx in citations:
                    if idx < len(docs):
                        doc = docs[idx]
                        metadata = doc.metadata
                        # Look for the right field names in metadata
                        doc_id = metadata.get('doc_id', metadata.get('id', f'doc-{idx}'))
                        doc_title = metadata.get('title', f'Document {idx}')
                        
                        formatted_answer += f"[{idx}] {doc_id}: {doc_title}\n"
                        
                        # Add page and author if available
                        if 'page' in metadata and metadata['page'] is not None:
                            formatted_answer += f"    Page: {metadata['page']}\n"
                        if 'author' in metadata and metadata['author'] is not None:
                            formatted_answer += f"    Author: {metadata['author']}\n"
            
            return formatted_answer, True if citations else False
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(traceback.format_exc())
            return f"Error processing query: {e}", False
    
    # Process each query
    for i, query in enumerate(queries):
        logger.info(f"\nProcessing query {i+1}: {query}")
        formatted_result, has_citations = process_query(query)
        
        # Log the result
        logger.info(f"Result:\n{formatted_result}")
        
        # Check if citations are included
        if has_citations:
            citation_success += 1
            logger.info(f"SUCCESS: Response included citations for query: '{query}'")
        else:
            logger.error(f"ERROR: Response did not include citations for query: '{query}'")
            
        # Add result to results list
        results.append({"query": query, "answer": formatted_result, "has_citations": has_citations})
    
    # Calculate and report success rate
    success_rate = (citation_success / total_queries) * 100 if total_queries > 0 else 0
    logger.info(f"\nCitation success rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        logger.info("SUCCESS: All responses included proper citations!")
    else:
        logger.error(f"ERROR: Only {citation_success} out of {total_queries} responses included proper citations.")
    
    # Clean up if client was provided
    if client:
        client.close()
    
    logger.info("Full RAG pipeline test completed!")
    return results

def main():
    """Run the full RAG pipeline test with Milvus."""
    logger.info("Starting test of full RAG pipeline with Milvus and citation functionality")
    
    try:
        # 1. Check OpenAI API key
        check_openai_api_key()
        
        # 2. Define test queries
        queries = [
            "What is machine learning?",
            "Explain deep learning and how it relates to neural networks.",
            "What are some applications of natural language processing?",
            "Compare supervised and unsupervised learning."
        ]
        logger.info(f"Running {len(queries)} test queries")
        
        # 3. Create the RAG pipeline
        rag_pipeline, retriever, client = create_rag_pipeline()
        
        # 4. Run the test queries
        results = run_test_queries(rag_pipeline, queries, client)
        
        # 5. Evaluate results
        citation_success = all(result["has_citations"] for result in results)
        if citation_success:
            logger.info("SUCCESS: All responses included proper citations!")
        else:
            logger.warning("Some responses did not include proper citations")
        
        return citation_success
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
