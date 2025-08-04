"""Enhanced retrieval service with query processing and ranking."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class RetrievalService:
    """Enhanced retrieval with better query processing and ranking."""
    
    def __init__(self, vector_store, embedder, query_processor):
        self.vector_store = vector_store
        self.embedder = embedder
        self.query_processor = query_processor

    def search_with_enhanced_query(self, query: str, top_k: int = 5, 
                                  code_context: str = None) -> List[Dict]:
        """Enhanced retrieval with better query processing."""
        # Enhance the query with context
        enhanced_query = self.query_processor.enhance_query_with_context(query, code_context)
        
        # Get embedding
        query_embedding = self.embedder.embed_query(enhanced_query)
        
        # Search with more candidates for better reranking
        candidates = self.vector_store.search_similar(query_embedding, top_k * 3)
        
        if not candidates:
            return []
        
        # Enhanced reranking based on multiple factors
        for candidate in candidates:
            base_score = candidate.get('score', 0)
            boost = 0.0
            metadata = candidate.get('metadata', {})
            
            # Boost for refactored examples
            if metadata.get('version') == 'refactored':
                boost += 0.15
            
            # Boost for matching intent
            query_lower = query.lower()
            refactoring_intents = metadata.get('refactoring_intents', [])
            
            if 'complex' in query_lower and 'complexity_reduction' in refactoring_intents:
                boost += 0.1
            if 'readable' in query_lower and 'readability_improvement' in refactoring_intents:
                boost += 0.1
            if 'performance' in query_lower and 'performance_optimization' in refactoring_intents:
                boost += 0.1
            
            # Boost for code type matching
            if 'function' in query_lower and metadata.get('type') == 'function':
                boost += 0.08
            if 'class' in query_lower and metadata.get('type') == 'class':
                boost += 0.08
            
            # Boost for complexity matching
            if any(word in query_lower for word in ['complex', 'nested', 'simplify']):
                complexity = metadata.get('complexity', {}).get('cyclomatic_complexity', 0)
                if complexity > 5:
                    boost += 0.1
            
            candidate['score'] = min(base_score + boost, 1.0)
        
        # Sort and return
        candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
        return candidates[:top_k]

    def search_by_metadata(self, query: str, filters: Dict[str, Any], 
                          top_k: int = 5) -> List[Dict]:
        """Search with metadata filtering."""
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search_with_filter(query_embedding, top_k, filters)

    def search_similar_code(self, code: str, top_k: int = 5) -> List[Dict]:
        """Search for similar code examples."""
        # Create a code-focused query
        enhanced_query = f"Python code example: {code}"
        query_embedding = self.embedder.embed_query(enhanced_query)
        
        results = self.vector_store.search_similar(query_embedding, top_k)
        
        # Filter to only code chunks
        code_results = [
            result for result in results 
            if result.get('metadata', {}).get('type') in ['function', 'class', 'code_block']
        ]
        
        return code_results[:top_k]

    def hybrid_search(self, query: str, code_context: str = None, 
                     top_k: int = 5) -> List[Dict]:
        """Combine semantic and keyword-based search."""
        # Get semantic results
        semantic_results = self.search_with_enhanced_query(query, top_k, code_context)
        
        # Simple keyword matching as fallback
        query_words = set(query.lower().split())
        
        # Re-rank based on keyword overlap
        for result in semantic_results:
            content = result.get('text', '').lower()
            content_words = set(content.split())
            overlap = len(query_words.intersection(content_words))
            
            # Add keyword boost
            keyword_boost = overlap / max(1, len(query_words)) * 0.1
            result['score'] = min(result.get('score', 0) + keyword_boost, 1.0)
        
        # Re-sort after keyword boosting
        semantic_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return semantic_results