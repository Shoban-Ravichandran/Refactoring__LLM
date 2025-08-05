"""Enhanced RAG service with PDF processing skip logic to avoid redundant embedding generation."""

import logging
import os
import hashlib
from typing import List, Dict, Any, Union, Optional, Set
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
    """Enhanced RAG system with intelligent PDF processing skip logic."""
    
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
        
        # Enhanced state management
        self.best_model: Optional[str] = None
        self.optimization_results: Optional[Dict] = None
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

    def set_best_model(self, model_name: str, optimization_results: Dict = None):
        """Set the best model based on optimization results."""
        if model_name in self.llm_provider.get_available_models():
            self.best_model = model_name
            self.optimization_results = optimization_results
            logger.info(f"Best model set to: {model_name}")
        else:
            logger.warning(f"Model {model_name} not available. Available models: {self.llm_provider.get_available_models()}")

    def get_best_model(self) -> Optional[str]:
        """Get the current best model."""
        return self.best_model

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

    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for a file to track if it has been processed."""
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return ""
        
        # Create hash from file path, size, and modification time
        stat_info = file_path_obj.stat()
        hash_input = f"{file_path}:{stat_info.st_size}:{stat_info.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _get_processed_pdfs(self) -> Set[str]:
        """Get set of already processed PDF hashes from vector store."""
        processed_pdfs = set()
        
        try:
            # Search for PDF chunks in the vector store
            offset = None
            while True:
                points, next_offset = self.vector_store.scroll_points(limit=100, offset=offset)
                
                if not points:
                    break
                
                for point in points:
                    metadata = point.get('metadata', {})
                    if metadata.get('content_type') == 'text' and 'source_file' in metadata:
                        # Look for PDF file hash in metadata
                        pdf_hash = metadata.get('pdf_file_hash')
                        if pdf_hash:
                            processed_pdfs.add(pdf_hash)
                
                offset = next_offset
                if offset is None:
                    break
                    
        except Exception as e:
            logger.warning(f"Error checking processed PDFs: {e}")
        
        return processed_pdfs

    def check_pdf_processing_status(self, pdf_paths: List[str]) -> Dict[str, bool]:
        """Check which PDFs have already been processed."""
        if not pdf_paths:
            return {}
        
        logger.info("Checking PDF processing status...")
        
        # Get already processed PDF hashes
        processed_pdf_hashes = self._get_processed_pdfs()
        
        pdf_status = {}
        for pdf_path in pdf_paths:
            if not Path(pdf_path).exists():
                pdf_status[pdf_path] = False  # File doesn't exist
                continue
            
            pdf_hash = self._get_file_hash(pdf_path)
            pdf_status[pdf_path] = pdf_hash in processed_pdf_hashes
            
            if pdf_status[pdf_path]:
                logger.info(f"PDF already processed: {Path(pdf_path).name}")
            else:
                logger.info(f"PDF needs processing: {Path(pdf_path).name}")
        
        return pdf_status

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

    def process_pdfs(self, pdf_paths: List[str], force_reindex: bool = False) -> int:
        """Process PDF files with intelligent skip logic to avoid redundant processing."""
        if not self._is_setup:
            raise RuntimeError("System not setup. Call setup() first.")
        
        valid_paths = [path for path in pdf_paths if Path(path).exists()]
        if not valid_paths:
            logger.warning("No valid PDF paths provided")
            return 0
        
        # Check which PDFs have already been processed
        if not force_reindex:
            pdf_status = self.check_pdf_processing_status(valid_paths)
            pdfs_to_process = [path for path, processed in pdf_status.items() if not processed]
            already_processed = [path for path, processed in pdf_status.items() if processed]
            
            if already_processed:
                logger.info(f"Skipping {len(already_processed)} already processed PDFs:")
                for pdf_path in already_processed:
                    logger.info(f"  - {Path(pdf_path).name}")
            
            if not pdfs_to_process:
                logger.info("All PDFs have already been processed. No new processing needed.")
                return 0
            
            logger.info(f"Processing {len(pdfs_to_process)} new/updated PDFs:")
            for pdf_path in pdfs_to_process:
                logger.info(f"  - {Path(pdf_path).name}")
            
            valid_paths = pdfs_to_process
        else:
            logger.info(f"Force reindex enabled. Processing all {len(valid_paths)} PDF files")
        
        all_chunks = []
        successfully_processed = []
        
        for pdf_path in valid_paths:
            try:
                logger.info(f"Processing PDF: {pdf_path}")
                text = self.pdf_processor.extract_text_from_pdf(pdf_path)
                
                if text:
                    # Generate hash for this PDF
                    pdf_hash = self._get_file_hash(pdf_path)
                    
                    pdf_metadata = {
                        'source_file': Path(pdf_path).name,
                        'source_path': pdf_path,
                        'content_type': 'text',
                        'pdf_file_hash': pdf_hash,  # Add hash to track processing
                        'processed_timestamp': str(Path.ctime(Path.cwd()))
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
                              f"{len(code_chunks)} code chunks from {Path(pdf_path).name}")
                    
                    successfully_processed.append(pdf_path)
                else:
                    logger.warning(f"No text extracted from {pdf_path}")
                    
            except Exception as e:
                logger.error(f"Failed to process PDF {pdf_path}: {e}")
                continue
        
        # Generate embeddings and store
        if all_chunks:
            logger.info(f"Generating embeddings for {len(all_chunks)} PDF chunks...")
            chunks_with_embeddings = self.embedder.embed_chunks(all_chunks)
            self.vector_store.insert_chunks(chunks_with_embeddings)
            logger.info(f"Successfully stored {len(all_chunks)} PDF chunks from {len(successfully_processed)} PDFs")
        
        return len(all_chunks)

    def force_reprocess_pdfs(self, pdf_paths: List[str] = None) -> int:
        """Force reprocessing of PDFs, removing old chunks first."""
        if not self._is_setup:
            raise RuntimeError("System not setup. Call setup() first.")
        
        if pdf_paths is None:
            logger.warning("No PDF paths specified for force reprocessing")
            return 0
        
        logger.info("Force reprocessing PDFs - removing old chunks first...")
        
        # Remove old PDF chunks
        removed_count = self._remove_pdf_chunks(pdf_paths)
        logger.info(f"Removed {removed_count} old PDF chunks")
        
        # Process PDFs with force reindex
        return self.process_pdfs(pdf_paths, force_reindex=True)

    def _remove_pdf_chunks(self, pdf_paths: List[str]) -> int:
        """Remove chunks from specific PDF files."""
        pdf_names = [Path(path).name for path in pdf_paths]
        removed_count = 0
        
        try:
            offset = None
            points_to_remove = []
            
            while True:
                points, next_offset = self.vector_store.scroll_points(limit=100, offset=offset)
                
                if not points:
                    break
                
                for i, point in enumerate(points):
                    metadata = point.get('metadata', {})
                    source_file = metadata.get('source_file', '')
                    
                    if source_file in pdf_names:
                        # Calculate point ID based on scroll position
                        point_id = i if offset is None else int(offset) + i
                        points_to_remove.append(point_id)
                
                offset = next_offset
                if offset is None:
                    break
            
            # Remove the identified points
            if points_to_remove:
                self.vector_store.delete_points(points_to_remove)
                removed_count = len(points_to_remove)
                
        except Exception as e:
            logger.error(f"Error removing PDF chunks: {e}")
        
        return removed_count

    def get_pdf_processing_summary(self) -> Dict[str, Any]:
        """Get summary of PDF processing status."""
        summary = {
            'total_chunks': 0,
            'pdf_files': {},
            'processing_dates': {}
        }
        
        try:
            offset = None
            while True:
                points, next_offset = self.vector_store.scroll_points(limit=100, offset=offset)
                
                if not points:
                    break
                
                for point in points:
                    metadata = point.get('metadata', {})
                    if metadata.get('content_type') == 'text' and 'source_file' in metadata:
                        source_file = metadata['source_file']
                        
                        if source_file not in summary['pdf_files']:
                            summary['pdf_files'][source_file] = {
                                'text_chunks': 0,
                                'code_chunks': 0,
                                'total_chunks': 0
                            }
                        
                        chunk_type = metadata.get('type', 'text')
                        if chunk_type == 'text':
                            summary['pdf_files'][source_file]['text_chunks'] += 1
                        else:
                            summary['pdf_files'][source_file]['code_chunks'] += 1
                        
                        summary['pdf_files'][source_file]['total_chunks'] += 1
                        summary['total_chunks'] += 1
                        
                        # Track processing timestamp
                        if 'processed_timestamp' in metadata:
                            summary['processing_dates'][source_file] = metadata['processed_timestamp']
                
                offset = next_offset
                if offset is None:
                    break
                    
        except Exception as e:
            logger.error(f"Error getting PDF summary: {e}")
        
        return summary

    def get_refactoring_suggestions(self, query: str,
                                  config: RAGConfig = None,
                                  model_name: str = None,
                                  user_code: str = None,
                                  use_best_model: bool = False) -> Union[str, Dict[str, str]]:
        """Enhanced method to get refactoring suggestions with flexible model selection."""
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
        
        # Determine which model(s) to use
        if use_best_model and self.best_model:
            model_name = self.best_model
        
        # Generate suggestions
        if model_name:
            # Single model response
            if model_name not in self.llm_provider.get_available_models():
                return f"Model '{model_name}' is not available. Available models: {self.llm_provider.get_available_models()}"
            
            suggestion = self.llm_provider.generate_suggestion(
                query, similar_chunks, config, model_name, user_code
            )
            return suggestion
        else:
            # Return suggestions from all models
            suggestions = {}
            available_models = self.llm_provider.get_available_models()
            
            # Prioritize best model if available
            if self.best_model and self.best_model in available_models:
                models_to_query = [self.best_model] + [m for m in available_models if m != self.best_model]
            else:
                models_to_query = available_models
            
            for model in models_to_query:
                try:
                    suggestion = self.llm_provider.generate_suggestion(
                        query, similar_chunks, config, model, user_code
                    )
                    suggestions[model] = suggestion
                except Exception as e:
                    logger.error(f"Error generating suggestion with {model}: {e}")
                    suggestions[model] = f"Error: {str(e)}"
            
            return suggestions

    def get_best_model_suggestion(self, query: str,
                                config: RAGConfig = None,
                                user_code: str = None) -> str:
        """Get suggestion from the best model only."""
        if not self.best_model:
            return "No best model has been determined. Run optimization first or set a best model manually."
        
        return self.get_refactoring_suggestions(
            query=query,
            config=config,
            model_name=self.best_model,
            user_code=user_code
        )

    def get_all_model_suggestions(self, query: str,
                                config: RAGConfig = None,
                                user_code: str = None) -> Dict[str, str]:
        """Get suggestions from all available models."""
        result = self.get_refactoring_suggestions(
            query=query,
            config=config,
            model_name=None,  # This will trigger all models
            user_code=user_code
        )
        
        if isinstance(result, dict):
            return result
        else:
            # Single model response (shouldn't happen with model_name=None)
            return {"single_model": result}

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics including PDF processing info."""
        if not self._is_setup:
            return {'status': 'not_setup'}
        
        stats = {
            'status': 'ready',
            'vector_store': self.vector_store.get_stats(),
            'embedder': self.embedder.get_embedding_stats(),
            'llm_models': self.llm_provider.get_available_models(),
            'best_model': self.best_model,
            'optimization_completed': self.optimization_results is not None,
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
        
        # Add PDF processing summary
        try:
            pdf_summary = self.get_pdf_processing_summary()
            stats['pdf_processing'] = pdf_summary
        except Exception as e:
            logger.warning(f"Error getting PDF summary: {e}")
            stats['pdf_processing'] = {'error': str(e)}
        
        if self.optimization_results:
            stats['optimization_metrics'] = {
                'algorithm': self.optimization_results.get('algorithm', 'Unknown'),
                'pareto_front_size': self.optimization_results.get('pareto_front_size', 0),
                'optimization_time': self.optimization_results.get('optimization_time_seconds', 0)
            }
        
        return stats

    def health_check(self) -> Dict[str, bool]:
        """Perform comprehensive health check on all components."""
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
            available_models = self.llm_provider.get_available_models()
            health['llm_provider'] = len(available_models) > 0
            
            # Test best model if set
            if self.best_model:
                health['best_model_available'] = self.best_model in available_models
            else:
                health['best_model_available'] = True  # Not set, so not a failure
                
        except Exception as e:
            health['llm_provider'] = False
            health['best_model_available'] = False
            logger.error(f"LLM provider health check failed: {e}")
        
        # Overall health
        health['overall'] = all(health.values())
        
        return health

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