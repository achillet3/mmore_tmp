# Enhanced Citation Functionality in RAG Pipeline

## Overview
This PR enhances the RAG pipeline's citation functionality by improving the default prompt to explicitly encourage citation inclusion. The changes ensure that LLM responses consistently include citations in their answers without requiring extra parameters or configuration.

## Changes
- Enhanced the DEFAULT_PROMPT in `src/mmore/rag/pipeline.py` to explicitly instruct the LLM to include citations using document references like [0], [1], etc.
- Added a targeted test file `tests/test_citations.py` that validates the citation formatting functionality

## Testing
The changes were thoroughly tested using:
- [X] Unit tests: A dedicated unit test (`tests/test_citations.py`) that validates the citation formatting functionality
- [X] End-to-end tests: The end-to-end pipeline with real Milvus data collection was tested locally. My test can be added if needed and if it makes sense (I thought it was a little bit too system dependent and might require some elaborate CI/CD pipeline)

The test confirms that document metadata is properly used in the citation block and that citation references are correctly included in the answer text.

## Implementation Details
- The solution modifies only the DEFAULT_PROMPT while leaving the rest of the implementation intact
- Citation markers [0], [1], etc. appear directly in the answer text
- Citations are properly formatted with the document ID and title
- No additional parameters are required to enable citations - they work by default

## Benefits
- **Consistency**: Citations are now part of the default behavior
- **Simplicity**: No special parameters needed for citation functionality
- **Reliability**: Tested with different query types and document sources
- **Maintainability**: Clean, focused implementation that builds on the existing architecture

## Dependency Note
Implementation is compatible with the existing LangChain dependency structure and works correctly with the current environment.

## Future Improvements

### Enhanced Metadata in Citations
Based on examining the test implementation with real Milvus data, here's how we could enhance citation metadata:

1. **Extract richer metadata in document processors**:
   ```python
   # In src/mmore/process/processors/pdf_processor.py
   def process(self, file_path: str) -> List[MultimodalSample]:
       # Use PyMuPDF to extract metadata
       pdf_doc = fitz.open(file_path)
       metadata = {
           "author": pdf_doc.metadata.get("author", ""),
           "publication_date": pdf_doc.metadata.get("creationDate", ""),
           "page_count": len(pdf_doc),
           "source": file_path
       }
       # Process text and create sample with metadata
       text = process_pdf_text(pdf_doc)
       return self.create_sample([text], [], file_path, metadata=metadata)
   ```

2. **Include metadata fields when inserting into Milvus**:
   ```python
   # When inserting documents into Milvus (no schema changes needed)
   entity = {
       "text": document.text,
       "dense_embedding": dense_vec,
       # Core citation fields (existing)
       "doc_id": document.id,
       "title": document.title,
       # Additional metadata fields (new)
       "author": document.metadata.get("author", ""),
       "publication_date": document.metadata.get("publication_date", ""),
       "page_count": document.metadata.get("page_count", ""),
       "source": document.metadata.get("source", "")
   }
   ```

3. **Request additional fields in the retriever**:
   ```python
   # In retriever implementation
   output_fields = ["doc_id", "title", "author", "publication_date", "page_count", "source"]
   ```

4. **Use the additional metadata in citation formatting**:
   ```python
   # In RAGPipeline.__call__ method
   if citations:
       formatted_answer += "\n\nCitations:\n"
       for citation_idx in citations:
           if citation_idx < len(docs):
               doc = docs[citation_idx]
               doc_id = doc.metadata.get('doc_id', doc.metadata.get('id', f'doc-{citation_idx}'))
               doc_title = doc.metadata.get('title', f'Document {citation_idx}')
               
               # Basic citation (existing)
               formatted_answer += f"[{citation_idx}] {doc_id}: {doc_title}\n"
               
               # Enhanced citation with additional metadata (new)
               if 'author' in doc.metadata and doc.metadata['author']:
                   formatted_answer += f"    Author: {doc.metadata['author']}\n"
               if 'publication_date' in doc.metadata and doc.metadata['publication_date']:
                   formatted_answer += f"    Published: {doc.metadata['publication_date']}\n"
               if 'source' in doc.metadata and doc.metadata['source']:
                   formatted_answer += f"    Source: {doc.metadata['source']}\n"
   ```

This approach leverages existing functionality for storing and retrieving metadata fields without requiring schema changes or complex citation formatting systems.
