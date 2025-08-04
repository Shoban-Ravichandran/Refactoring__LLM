"""Code-specific embedding models and utilities."""

import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")


class CodeEmbedder:
    """Enhanced embedder with better context creation for code."""
    
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-code"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for CodeEmbedder")
        
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model with fallback options."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded code-specific embedding model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}")
            logger.warning("Trying alternative models...")
            
            alternatives = [
                "microsoft/codebert-base",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/all-MiniLM-L6-v2"
            ]
            
            for alt_model in alternatives:
                try:
                    self.model = SentenceTransformer(alt_model)
                    self.model_name = alt_model
                    logger.info(f"Loaded alternative embedding model: {alt_model}")
                    break
                except Exception as alt_e:
                    logger.warning(f"Failed to load {alt_model}: {alt_e}")
                    continue
            
            if self.model is None:
                raise RuntimeError("Could not load any suitable embedding model")
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def create_enhanced_text_for_embedding(self, chunk: Dict) -> str:
        """Create better text representation for embeddings."""
        code = chunk.get('code', chunk.get('text', ''))
        metadata = chunk.get('metadata', {})
        
        # Build context-rich text for embedding
        text_parts = []
        
        # Add semantic context
        chunk_type = metadata.get('type', 'code')
        if chunk_type == 'function':
            name = metadata.get('name', 'unnamed')
            text_parts.append(f"Python function {name}")
            
            # Add complexity context
            complexity = metadata.get('complexity', {}).get('cyclomatic_complexity', 0)
            if complexity > 8:
                text_parts.append("high complexity function")
            elif complexity <= 3:
                text_parts.append("simple function")
                
        elif chunk_type == 'class':
            name = metadata.get('name', 'unnamed')
            text_parts.append(f"Python class {name}")
        
        # Add refactoring context  
        version = metadata.get('version')
        if version == 'refactored':
            text_parts.append("improved refactored code example")
        elif version == 'original':
            text_parts.append("original code before refactoring")
        
        # Add pattern context
        refactoring_type = metadata.get('refactoring_type', '')
        if refactoring_type:
            clean_type = refactoring_type.replace('_', ' ')
            text_parts.append(f"{clean_type} refactoring pattern")
        
        # Add intent context
        refactoring_intents = metadata.get('refactoring_intents', [])
        if refactoring_intents:
            text_parts.extend([intent.replace('_', ' ') for intent in refactoring_intents])
        
        # Combine context with code
        if text_parts:
            context = ' | '.join(text_parts)
            enhanced_text = f"{context}\n{code}"
        else:
            enhanced_text = code
        
        return enhanced_text

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings with enhanced text."""
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        enhanced_texts = []
        for chunk in chunks:
            enhanced_text = self.create_enhanced_text_for_embedding(chunk)
            enhanced_texts.append(enhanced_text)
        
        # Generate embeddings in batches
        try:
            embeddings = self.model.encode(
                enhanced_texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = embeddings[i].tolist()
            
            logger.info(f"Successfully generated embeddings for {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        
        return chunks

    def embed_query(self, query: str) -> np.ndarray:
        """Generate query embedding with Python context."""
        enhanced_query = f"Python code refactoring: {query}"
        
        try:
            embedding = self.model.encode(
                [enhanced_query],
                normalize_embeddings=True,
                convert_to_numpy=True
            )[0]
            
            logger.debug(f"Query embedded: '{query}' -> '{enhanced_query}'")
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Reshape if needed
            if embedding1.ndim == 1:
                embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1:
                embedding2 = embedding2.reshape(1, -1)
            
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
            
        except ImportError:
            # Fallback to manual cosine similarity calculation
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))

    def find_most_similar_chunks(self, query_embedding: np.ndarray, 
                                chunk_embeddings: List[np.ndarray], 
                                top_k: int = 5) -> List[tuple]:
        """Find most similar chunks to a query embedding."""
        similarities = []
        
        for i, chunk_embedding in enumerate(chunk_embeddings):
            similarity = self.calculate_similarity(query_embedding, chunk_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

    def batch_embed_texts(self, texts: List[str], 
                          batch_size: int = 32,
                          show_progress: bool = True) -> np.ndarray:
        """Embed a batch of texts efficiently."""
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            raise

    def embed_code_with_context(self, code: str, 
                               context_type: str = 'refactoring',
                               additional_context: str = '') -> np.ndarray:
        """Embed code with specific context for better retrieval."""
        context_templates = {
            'refactoring': f"Python code refactoring example: {additional_context}",
            'optimization': f"Python code optimization example: {additional_context}",
            'debugging': f"Python code debugging example: {additional_context}",
            'pattern': f"Python design pattern example: {additional_context}",
            'general': f"Python code example: {additional_context}"
        }
        
        context = context_templates.get(context_type, context_templates['general'])
        enhanced_text = f"{context}\n{code}" if additional_context else f"{context}\n{code}"
        
        return self.model.encode([enhanced_text], normalize_embeddings=True, convert_to_numpy=True)[0]

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding model."""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'max_sequence_length': getattr(self.model, 'max_seq_length', 'unknown'),
            'model_type': type(self.model).__name__
        }

    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to file."""
        try:
            np.save(filepath, embeddings)
            logger.info(f"Embeddings saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise

    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file."""
        try:
            embeddings = np.load(filepath)
            logger.info(f"Embeddings loaded from {filepath}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise

    def validate_embedding_dimension(self, embedding: np.ndarray) -> bool:
        """Validate that embedding has correct dimension."""
        if embedding.ndim == 1:
            return len(embedding) == self.embedding_dim
        elif embedding.ndim == 2:
            return embedding.shape[1] == self.embedding_dim
        else:
            return False


class EmbeddingCache:
    """Simple cache for embeddings to avoid recomputation."""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get(self, text: str) -> np.ndarray:
        """Get embedding from cache."""
        text_hash = hash(text)
        if text_hash in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(text_hash)
            self.access_order.append(text_hash)
            return self.cache[text_hash]
        return None
    
    def put(self, text: str, embedding: np.ndarray):
        """Put embedding in cache."""
        text_hash = hash(text)
        
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and text_hash not in self.cache:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[text_hash] = embedding
        if text_hash in self.access_order:
            self.access_order.remove(text_hash)
        self.access_order.append(text_hash)
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


class CachedCodeEmbedder(CodeEmbedder):
    """Code embedder with caching for improved performance."""
    
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-code",
                 cache_size: int = 1000):
        super().__init__(model_name)
        self.cache = EmbeddingCache(cache_size)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate query embedding with caching."""
        enhanced_query = f"Python code refactoring: {query}"
        
        # Check cache first
        cached_embedding = self.cache.get(enhanced_query)
        if cached_embedding is not None:
            logger.debug(f"Using cached embedding for query: {query}")
            return cached_embedding
        
        # Generate new embedding
        embedding = super().embed_query(query)
        
        # Cache the result
        self.cache.put(enhanced_query, embedding)
        
        return embedding
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings with caching."""
        uncached_chunks = []
        uncached_indices = []
        
        # Check cache for each chunk
        for i, chunk in enumerate(chunks):
            enhanced_text = self.create_enhanced_text_for_embedding(chunk)
            cached_embedding = self.cache.get(enhanced_text)
            
            if cached_embedding is not None:
                chunk['embedding'] = cached_embedding.tolist()
            else:
                uncached_chunks.append(chunk)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached chunks
        if uncached_chunks:
            logger.info(f"Generating embeddings for {len(uncached_chunks)} uncached chunks")
            enhanced_texts = [self.create_enhanced_text_for_embedding(chunk) for chunk in uncached_chunks]
            
            embeddings = self.model.encode(
                enhanced_texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            
            # Add embeddings and cache them
            for i, (chunk_idx, chunk) in enumerate(zip(uncached_indices, uncached_chunks)):
                embedding = embeddings[i]
                chunk['embedding'] = embedding.tolist()
                
                enhanced_text = enhanced_texts[i]
                self.cache.put(enhanced_text, embedding)
        
        logger.info(f"Embeddings generated for {len(chunks)} chunks ({len(uncached_chunks)} new, {len(chunks) - len(uncached_chunks)} cached)")
        
        return chunks
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': self.cache.size(),
            'max_cache_size': self.cache.max_size,
            'cache_hit_rate': 'Not tracked'  # Could implement hit rate tracking if needed
        }