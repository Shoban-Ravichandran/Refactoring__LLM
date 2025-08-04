"""Main RAG service orchestrating the complete refactoring system."""

import logging
import os
from typing import List, Dict, Any, Union, Optional
from pathlib import Path

from config.settings import RAGConfig, PDFConfig, get_default_config
from config.model_configs import LLMConfig
from data.embeddings.code_embedder import CodeEmbedder
from data.processors.pdf_processor import PDFProcessor
from data.processors.code_chunker import CodeChunker
from services.vector_store import QdrantVectorStore
from services.retrieval_service import RetrievalService
from services.query_processor import QueryProcessor
from models.llm_providers import MultiLLMProvider

logger = logging.getLogger(__name__)


class RefactoringRAGSystem:
    """Complete RAG system for Python code refactoring."""
    
    def __init__(self, llm_configs: List[LLMConfig],
                 qdrant_url: str = None,
                 qdrant_api_key: str = None):
        self.llm_configs = llm_configs
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        
        # Initialize components
        self.embedder: Optional[CodeEmbedder] = None
        self.vector_store: Optional[QdrantVectorStore] = None
        self.retrieval_service: Optional[RetrievalService] = None
        self.query_processor: Optional[QueryProcessor] = None
        self.llm_provider: Optional[MultiLLMProvider] = None
        self.pdf_processor: Optional[PDFProcessor] = None
        self.code_chunker: Optional[CodeChunker] = None
        
        self._is_setup = False

    def setup(self, embedding_model: str = "jinaai/jina-embeddings-v2-base-code",
              config: Dict[str, Any] = None):
        """Initialize all components."""
        if self._is_setup:
            logger.info("System already setup")
            return
        
        if config is None:
            config = get_default_config()
        
        logger.info("Setting up RefactoringRAGSystem...")
        
        # Initialize embedder
        self.embedder = CodeEmbedder(embedding_model)
        logger.info(f"Initialized embedder with dimension: {self.embedder.embedding_dim}")
        
        # Initialize vector store
        if self.qdrant_url:
            self.vector_store = QdrantVectorStore(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                collection_name="code_refactoring_chunks"
            )
        else:
            self.vector_store = QdrantVectorStore(collection_name="code_refactoring_chunks")
        
        # Create collection
        self.vector_store.create_collection(self.embedder.embedding_dim)
        
        # Initialize other components
        self.query_processor = QueryProcessor()
        self.retrieval_service = RetrievalService(self.vector_store, self.embedder, self.query_processor)
        self.llm_provider = MultiLLMProvider(self.llm_configs)
        self.pdf_processor = PDFProcessor(config.get('pdf', PDFConfig()))
        self.code_chunker = CodeChunker(config.get('code_chunk', config['code_chunk']))
        
        # Test connections
        self._test_connections()
        
        self._is_setup = True
        logger.info("RefactoringRAGSystem setup complete")

    def _test_connections(self):
        """Test that all connections are working."""
        # Test vector store
        info = self.vector_store.get_collection_info()
        if info:
            logger.info("Vector store connection successful")
        else:
            logger.error("Vector store connection failed")
        
        # Test LLM provider
        available_models = self.llm_provider.get_available_models()
        logger.info(f"Available LLM models: {available_models}")

    def process_dataset(self, dataset_path: str, force_reindex: bool = False) -> int:
        """Process dataset and add to vector store."""
        if not self._is_setup:
            raise RuntimeError("System not setup. Call setup() first.")
        
        # Check if collection already has data
        point_count = self.vector_store.count_points()
        if point_count > 0 and not force_reindex:
            logger.info(f"Collection already contains {point_count} points. Skipping dataset processing.")
            return point_count
        
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        logger.info(f"Processing dataset: {dataset_path}")
        
        # Load and process dataset
        import json
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        all_chunks = []
        processed = 0
        
        for i, item in enumerate(dataset):
            if processed >= 2500:  # Limit processing
                break
            
            original_code = item.get('original_code', '').strip()
            refactored_code = item.get('refactored_code', '').strip()
            
            # Skip invalid items
            if not original_code or not refactored_code:
                continue
            if len(original_code) < 40 or len(refactored_code) < 40:
                continue
            
            metadata = {
                'dataset_index': i,
                'description': item.get('description', ''),
                'refactoring_type': item.get('refactoring_type', ''),
                'language': 'python',
                'source': 'dataset'
            }
            
            try:
                # Process original code
                orig_chunks = self.code_chunker.process_code_aggressively(
                    original_code, {**metadata, 'version': 'original'}
                )
                all_chunks.extend(orig_chunks)
                
                # Process refactored code
                ref_chunks = self.code_chunker.process_code_aggressively(
                    refactored_code, {**metadata, 'version': 'refactored'}
                )
                all_chunks.extend(ref_chunks)
                
                processed += 1
                
            except Exception as e:
                logger.warning(f"Error processing example {i}: {e}")
                continue
        
        logger.info(f"Created {len(all_chunks)} chunks from {processed} examples")
        
        # Generate embeddings and store
        if all_chunks:
            chunks_with_embeddings = self.embedder.embed_chunks(all_chunks)
            self.vector_store.insert_chunks(chunks_with_embeddings)
            logger.info("Successfully stored chunks in vector database")
        
        return len(all_chunks)

    def process_pdfs(self, pdf_paths: List[str]) -> int:
        """Process PDF files and add to vector store."""
        if not self._is_setup:
            raise RuntimeError("System not setup. Call setup() first.")
        
        valid_paths = [path for path in pdf_paths if Path(path).exists()]
        if not valid_paths:
            logger.warning("No valid PDF paths provided")
            return 0
        
        logger.info(f"Processing {len(valid_paths)} PDF files")
        
        all_chunks = []
        for pdf_path in valid_paths:
            try:
                logger.info(f"Processing PDF: {pdf_path}")
                text = self.pdf_processor.extract_text_from_pdf(pdf_path)
                
                if text:
                    pdf_metadata = {
                        'source_file': Path(pdf_path).name,
                        'source_path': pdf_path,
                        'content_type': 'text'
                    }
                    
                    # Extract text chunks
                    text_chunks = self.pdf_processor.chunk_pdf_text(text, pdf_metadata)
                    all_chunks.extend(text_chunks)
                    
                    # Extract code blocks if enabled
                    code_chunks = self.pdf_processor.extract_code_blocks_from_text(text)
                    for chunk in code_chunks:
                        chunk['metadata'].update(pdf_metadata)
                    all_chunks.extend(code_chunks)
                    
                    logger.info(f"Extracted {len(text_chunks)} text chunks and "
                              f"{len(code_chunks)} code chunks from {pdf_path}")
                else:
                    logger.warning(f"No text extracted from {pdf_path}")
                    
            except Exception as e:
                logger.error(f"Failed to process PDF {pdf_path}: {e}")
                continue
        
        # Generate embeddings and store
        if all_chunks:
            chunks_with_embeddings = self.embedder.embed_chunks(all_chunks)
            self.vector_store.insert_chunks(chunks_with_embeddings)
            logger.info(f"Successfully stored {len(all_chunks)} PDF chunks")
        
        return len(all_chunks)

    def get_refactoring_suggestions(self, query: str,
                                  config: RAGConfig = None,
                                  model_name: str = None,
                                  user_code: str = None) -> Union[str, Dict[str, str]]:
        """Get refactoring suggestions for a query."""
        if not self._is_setup:
            raise RuntimeError("System not setup. Call setup() first.")
        
        if config is None:
            config = RAGConfig()
        
        logger.info(f"Getting refactoring suggestions for: {query[:100]}...")
        
        # Retrieve relevant chunks
        similar_chunks = self.retrieval_service.search_with_enhanced_query(
            query, config.top_k, user_code
        )
        
        if not similar_chunks:
            return "No similar code examples found. Please try a different query."
        
        # Log retrieval quality
        if similar_chunks:
            avg_score = sum(chunk.get('score', 0) for chunk in similar_chunks) / len(similar_chunks)
            logger.info(f"Retrieval quality: avg_score={avg_score:.3f}, chunks={len(similar_chunks)}")
        
        # Generate suggestions
        if model_name:
            suggestion = self.llm_provider.generate_suggestion(
                query, similar_chunks, config, model_name, user_code
            )
            return suggestion
        else:
            # Return suggestions from all models
            suggestions = {}
            for model in self.llm_provider.get_available_models():
                try:
                    suggestion = self.llm_provider.generate_suggestion(
                        query, similar_chunks, config, model, user_code
                    )
                    suggestions[model] = suggestion
                except Exception as e:
                    logger.error(f"Error generating suggestion with {model}: {e}")
                    suggestions[model] = f"Error: {str(e)}"
            
            return suggestions

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        if not self._is_setup:
            return {'status': 'not_setup'}
        
        stats = {
            'status': 'ready',
            'vector_store': self.vector_store.get_stats(),
            'embedder': self.embedder.get_embedding_stats(),
            'llm_models': self.llm_provider.get_available_models(),
            'components_initialized': {
                'embedder': self.embedder is not None,
                'vector_store': self.vector_store is not None,
                'retrieval_service': self.retrieval_service is not None,
                'query_processor': self.query_processor is not None,
                'llm_provider': self.llm_provider is not None,
                'pdf_processor': self.pdf_processor is not None,
                'code_chunker': self.code_chunker is not None
            }
        }
        
        return stats

    def clear_vector_store(self):
        """Clear all data from the vector store."""
        if not self._is_setup:
            raise RuntimeError("System not setup. Call setup() first.")
        
        self.vector_store.clear_collection()
        logger.info("Vector store cleared")

    def export_chunks(self, output_path: str, limit: int = None):
        """Export chunks from vector store to file."""
        if not self._is_setup:
            raise RuntimeError("System not setup. Call setup() first.")
        
        logger.info(f"Exporting chunks to {output_path}")
        
        chunks = []
        offset = None
        collected = 0
        
        while True:
            batch, next_offset = self.vector_store.scroll_points(limit=100, offset=offset)
            
            if not batch:
                break
            
            chunks.extend(batch)
            collected += len(batch)
            
            if limit and collected >= limit:
                chunks = chunks[:limit]
                break
            
            offset = next_offset
            if offset is None:
                break
        
        # Save to file
        import json
        with open(output_path, 'w') as f:
            json.dump(chunks, f, indent=2, default=str)
        
        logger.info(f"Exported {len(chunks)} chunks to {output_path}")

    def health_check(self) -> Dict[str, bool]:
        """Perform health check on all components."""
        health = {}
        
        if not self._is_setup:
            return {'system': False, 'error': 'System not setup'}
        
        try:
            # Test vector store
            health['vector_store'] = self.vector_store.get_collection_info() is not None
        except Exception as e:
            health['vector_store'] = False
            logger.error(f"Vector store health check failed: {e}")
        
        try:
            # Test embedder
            test_embedding = self.embedder.embed_query("test query")
            health['embedder'] = test_embedding is not None and len(test_embedding) > 0
        except Exception as e:
            health['embedder'] = False
            logger.error(f"Embedder health check failed: {e}")
        
        try:
            # Test LLM provider
            health['llm_provider'] = len(self.llm_provider.get_available_models()) > 0
        except Exception as e:
            health['llm_provider'] = False
            logger.error(f"LLM provider health check failed: {e}")
        
        health['overall'] = all(health.values())
        
        return health