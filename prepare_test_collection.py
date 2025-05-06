#!/usr/bin/env python3
"""
Script to create a test collection in Milvus and populate it with test documents.
This serves as a setup for testing the RAG pipeline with citation functionality.
"""

import logging
import sys
import os
from pathlib import Path
import numpy as np
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[MILVUS_SETUP] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Test documents with metadata
TEST_DOCS = [
    {
        "text": """Machine learning (ML) is a field of study that gives computers the ability to learn 
        without being explicitly programmed. ML algorithms build models based on sample data in order to 
        make predictions or decisions. The key aspect of machine learning is that the algorithms 
        improve automatically through experience.""",
        "metadata": {
            "document_id": "doc_001",
            "document_title": "Introduction to Machine Learning",
            "document_type": "textbook",
            "page": 1,
            "author": "Jane Smith"
        }
    },
    {
        "text": """Deep learning is a subset of machine learning that uses neural networks with many layers 
        (deep neural networks). These networks are designed to simulate the way the human brain works, with 
        each layer transforming the data and extracting increasingly complex features. Deep learning has been 
        successful in areas such as image and speech recognition, natural language processing, and autonomous driving.""",
        "metadata": {
            "document_id": "doc_002",
            "document_title": "Deep Learning Fundamentals",
            "document_type": "research_paper",
            "page": 15,
            "author": "John Doe"
        }
    },
    {
        "text": """Natural Language Processing (NLP) is a field of AI that focuses on the interaction 
        between computers and human language. NLP enables computers to understand, interpret, and generate 
        human language. Applications of NLP include machine translation, sentiment analysis, text summarization, 
        question answering, and chatbots.""",
        "metadata": {
            "document_id": "doc_003",
            "document_title": "NLP Techniques",
            "document_type": "academic_journal",
            "page": 7,
            "author": "Alex Johnson"
        }
    },
    {
        "text": """Supervised learning is a machine learning paradigm where the algorithm is trained 
        on labeled data, meaning the input data comes with the corresponding correct outputs. The algorithm 
        learns to map inputs to outputs based on example input-output pairs. Examples include classification 
        (e.g., spam detection) and regression (e.g., predicting house prices).""",
        "metadata": {
            "document_id": "doc_004",
            "document_title": "Machine Learning Paradigms",
            "document_type": "course_material",
            "page": 22,
            "author": "Michael Brown"
        }
    },
    {
        "text": """Unsupervised learning is a type of machine learning where the algorithm is given 
        unlabeled data and must find patterns or structure in the data on its own. The algorithm learns 
        from the data without explicit guidance. Examples include clustering (e.g., customer segmentation) 
        and dimensionality reduction (e.g., principal component analysis).""",
        "metadata": {
            "document_id": "doc_004",
            "document_title": "Machine Learning Paradigms",
            "document_type": "course_material",
            "page": 23,
            "author": "Michael Brown"
        }
    },
]

def create_test_vectors(doc_count, dim=384):
    """Create random test vectors for documents.
    
    Note: We're using 384 as the dimension for all-MiniLM-L6-v2 model.
    """
    """Create random test vectors for documents."""
    # Generate random vectors
    dense_vecs = []
    
    for i in range(doc_count):
        # Dense vector - simulate a sentence embedding
        dense_vec = np.random.rand(dim).astype(np.float32)
        # Normalize to unit length (cosine similarity)
        dense_vec = dense_vec / np.linalg.norm(dense_vec)
        dense_vecs.append(dense_vec.tolist())
    
    return dense_vecs

def create_collection(collection_name="test_collection"):
    """Create a collection in Milvus with the appropriate schema."""
    logger.info(f"Creating collection '{collection_name}'")
    
    # Define collection schema - Milvus standalone only supports one vector field
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4000),
        FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Use 384 for all-MiniLM-L6-v2
        # Include all metadata fields needed for citation
        # NOTE: Using 'id' and 'title' to match what the pipeline expects in pipeline.py
        # While also keeping the original document_id and document_title for reference
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),  # This is what pipeline.py looks for
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=100),   # This is what pipeline.py looks for
        FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),
        # Fields to store model configuration as strings since we can't have multiple vector fields
        FieldSchema(name="dense_model_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="sparse_model_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="is_multimodal", dtype=DataType.VARCHAR, max_length=10),
    ]
    schema = CollectionSchema(fields)
    
    # Create collection
    collection = Collection(name=collection_name, schema=schema)
    
    # Create index on vector fields
    logger.info("Creating indexes")
    
    # Create index on the vector field with the model metadata
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 8, "efConstruction": 64},
        # Add model metadata that will be used by the retriever
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "is_multimodal": "False"
    }
    collection.create_index(field_name="dense_embedding", index_params=index_params)
    
    return collection

def insert_documents(collection, documents):
    """Insert documents into the collection."""
    logger.info(f"Inserting {len(documents)} documents into collection")
    
    # Generate vector embeddings (using mock data for testing)
    dense_vecs = create_test_vectors(len(documents))
    
    # Prepare data in entity-by-entity format rather than field-by-field
    entities = []
    for i in range(len(documents)):
        entity = {
            "text": documents[i]["text"],
            "dense_embedding": dense_vecs[i],
            # Add both the doc_id and title fields that pipeline.py expects for citations
            "doc_id": documents[i]["metadata"]["document_id"],  # Pipeline looks for 'id' or 'doc_id'
            "title": documents[i]["metadata"]["document_title"],  # Pipeline looks for 'title'
            # Also keep the original document_* fields for reference
            "document_type": documents[i]["metadata"]["document_type"],
            "page": documents[i]["metadata"]["page"],
            "author": documents[i]["metadata"]["author"],
            # Store model configuration as strings
            "dense_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "sparse_model_name": "splade",
            "is_multimodal": "False",
        }
        entities.append(entity)
    
    # Insert data
    insert_result = collection.insert(entities)
    logger.info(f"Inserted {insert_result.insert_count} documents")
    
    # Flush to ensure data is persisted
    collection.flush()
    logger.info("Data flushed to storage")
    
    return insert_result.insert_count

def main():
    """Create test collection and insert documents."""
    logger.info("Starting Milvus test collection setup")
    
    collection_name = "test_collection"
    
    try:
        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530")
        logger.info("Connected to Milvus server")
        
        # Drop existing collection if it exists
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logger.info(f"Dropped existing collection '{collection_name}'")
        
        # Create new collection
        collection = create_collection(collection_name)
        
        # Insert test documents
        insert_count = insert_documents(collection, TEST_DOCS)
        
        # Load collection for searching
        collection.load()
        logger.info(f"Collection '{collection_name}' loaded with {insert_count} documents")
        
        # Simple search test to verify
        search_params = {"metric_type": "COSINE", "params": {"ef": 32}}
        results = collection.search(
            data=[create_test_vectors(1)[0]],  # Just use a random vector
            anns_field="dense_embedding",
            param=search_params,
            limit=2,
            output_fields=["text", "title"]  # Use the new field name 'title' instead of 'document_title'
        )
        
        logger.info(f"Verification search returned {len(results)} results")
        for i, hits in enumerate(results):
            logger.info(f"Top {len(hits)} documents for query {i+1}:")
            for hit in hits:
                logger.info(f"  - {hit.entity.title} (distance: {hit.distance})")
        
        logger.info("Test collection setup completed successfully!")
        return True
    
    except Exception as e:
        logger.error(f"Error during setup: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        # Disconnect from Milvus
        connections.disconnect("default")
        logger.info("Disconnected from Milvus server")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
