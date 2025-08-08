"""Vector storage using Qdrant for similarity search."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from qdrant_client.http import models as rest
    QDRANT_AVAILABLE = True
except ImportError:
    logger.error("qdrant-client not available. Install with: pip install qdrant-client")
    QDRANT_AVAILABLE = False


class QdrantVectorStore:
    """Vector storage using Qdrant Cloud or Local."""
    
    def __init__(self,
                 host: str = "localhost",
                 port: int = 6333,
                 collection_name: str = "code_points",
                 url: Optional[str] = None,
                 api_key: Optional[str] = None):
        """Initialize Qdrant client."""
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required for QdrantVectorStore")
        
        if url:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
            logger.info(f"Connected to Qdrant Cloud: {url}")
        else:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Connected to local Qdrant: {host}:{port}")
        
        self.collection_name = collection_name

    def create_collection(self, embedding_dim: int):
        """Create collection for storing embeddings."""
        try:
            collections = self.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if self.collection_name in existing_collections:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )
            logger.info(f"Created collection '{self.collection_name}' with dimension {embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error with collection: {e}")
            raise

    def insert_chunks(self, chunks: List[Dict]):
        """Insert chunks with embeddings into Qdrant."""
        if not chunks:
            logger.warning("No chunks to insert")
            return
        
        points = []
        valid_chunks = 0
        
        for i, chunk in enumerate(chunks):
            if 'embedding' not in chunk:
                logger.warning(f"Chunk {i} missing embedding")
                continue
            
            try:
                point = PointStruct(
                    id=i,
                    vector=chunk['embedding'],
                    payload={
                        'text': chunk['text'][:10000],  # Limit payload size
                        'code': chunk['code'][:10000],
                        'metadata': chunk['metadata'],
                        'chunk_id': chunk['chunk_id']
                    }
                )
                points.append(point)
                valid_chunks += 1
                
            except Exception as e:
                logger.error(f"Error creating point for chunk {i}: {e}")
                continue
        
        if not points:
            logger.error("No valid points to insert")
            return
        
        # Insert in batches
        batch_size = 200
        total_inserted = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.client.upsert(collection_name=self.collection_name, points=batch)
                total_inserted += len(batch)
                logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} points")
                
            except Exception as e:
                logger.error(f"Error inserting batch {i//batch_size + 1}: {e}")
        
        logger.info(f"Successfully inserted {total_inserted}/{valid_chunks} chunks into Qdrant")

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks using the updated API."""
        try:
            # Ensure query_embedding is the right format
            if isinstance(query_embedding, np.ndarray):
                query_vector = query_embedding.tolist()
            else:
                query_vector = query_embedding
            
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    timeout=30
                )
            
            similar_chunks = []
            for result in results:
                chunk_data = result.payload.copy()
                chunk_data['score'] = result.score
                similar_chunks.append(chunk_data)
            
            logger.debug(f"Found {len(similar_chunks)} similar chunks")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def search_with_filter(self, query_embedding: np.ndarray, 
                          top_k: int = 5,
                          filters: Dict[str, Any] = None) -> List[Dict]:
        """Search for similar chunks with metadata filtering."""
        try:
            query_vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            # Build filter conditions
            filter_conditions = None
            if filters:
                filter_conditions = rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key=f"metadata.{key}",
                            match=rest.MatchValue(value=value)
                        )
                        for key, value in filters.items()
                    ]
                )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=filter_conditions,
                limit=top_k,
                timeout=30
            )
            
            similar_chunks = []
            for result in results:
                chunk_data = result.payload.copy()
                chunk_data['score'] = result.score
                similar_chunks.append(chunk_data)
            
            logger.debug(f"Found {len(similar_chunks)} filtered similar chunks")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error in filtered search: {e}")
            return []

    def get_collection_info(self):
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            logger.debug(f"Collection info: {info}")
            return info
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None

    def delete_collection(self):
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def count_points(self) -> int:
        """Count the number of points in the collection."""
        try:
            info = self.get_collection_info()
            return info.points_count if info else 0
        except Exception as e:
            logger.error(f"Error counting points: {e}")
            return 0

    def get_point_by_id(self, point_id: int) -> Optional[Dict]:
        """Get a specific point by ID."""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False
            )
            
            if result:
                return result[0].payload
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving point {point_id}: {e}")
            return None

    def update_point_payload(self, point_id: int, payload_updates: Dict[str, Any]):
        """Update the payload of a specific point."""
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload_updates,
                points=[point_id]
            )
            logger.debug(f"Updated payload for point {point_id}")
            
        except Exception as e:
            logger.error(f"Error updating point {point_id}: {e}")

    def delete_points(self, point_ids: List[int]):
        """Delete specific points from the collection."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest.PointIdsList(points=point_ids)
            )
            logger.info(f"Deleted {len(point_ids)} points")
            
        except Exception as e:
            logger.error(f"Error deleting points: {e}")

    def scroll_points(self, limit: int = 100, offset: Optional[str] = None) -> tuple:
        """Scroll through points in the collection."""
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points = [point.payload for point in result[0]]
            next_offset = result[1]
            
            return points, next_offset
            
        except Exception as e:
            logger.error(f"Error scrolling points: {e}")
            return [], None

    def clear_collection(self):
        """Clear all points from the collection."""
        try:
            # Delete all points by scrolling and deleting in batches
            offset = None
            while True:
                points, next_offset = self.scroll_points(limit=1000, offset=offset)
                
                if not points:
                    break
                
                # Extract point IDs 
                point_ids = []
                for i, point in enumerate(points):
                    # Calculate point ID based on scroll position
                    point_id = i if offset is None else int(offset) + i
                    point_ids.append(point_id)
                
                if point_ids:
                    self.delete_points(point_ids)
                
                offset = next_offset
                if offset is None:
                    break
            
            logger.info(f"Cleared all points from collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector store."""
        try:
            info = self.get_collection_info()
            
            if not info:
                return {}
            
            stats = {
                'collection_name': self.collection_name,
                'points_count': info.points_count,
                'indexed_vectors_count': getattr(info, 'indexed_vectors_count', 0),
                'vector_size': info.config.params.vectors.size if hasattr(info.config, 'params') else 'unknown',
                'distance_metric': info.config.params.vectors.distance.value if hasattr(info.config, 'params') else 'unknown',
                'status': info.status.value if hasattr(info, 'status') else 'unknown'
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


class VectorStoreManager:
    """Manager for multiple vector stores or advanced operations."""
    
    def __init__(self):
        self.stores: Dict[str, QdrantVectorStore] = {}
    
    def add_store(self, name: str, store: QdrantVectorStore):
        """Add a vector store with a name."""
        self.stores[name] = store
        logger.info(f"Added vector store: {name}")
    
    def get_store(self, name: str) -> Optional[QdrantVectorStore]:
        """Get a vector store by name."""
        return self.stores.get(name)
    
    def search_multiple_stores(self, query_embedding: np.ndarray, 
                              store_names: List[str] = None,
                              top_k: int = 5) -> Dict[str, List[Dict]]:
        """Search across multiple vector stores."""
        if store_names is None:
            store_names = list(self.stores.keys())
        
        results = {}
        
        for store_name in store_names:
            store = self.stores.get(store_name)
            if store:
                try:
                    store_results = store.search_similar(query_embedding, top_k)
                    results[store_name] = store_results
                except Exception as e:
                    logger.error(f"Error searching store {store_name}: {e}")
                    results[store_name] = []
            else:
                logger.warning(f"Store {store_name} not found")
                results[store_name] = []
        
        return results
    
    def get_combined_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all managed stores."""
        stats = {}
        
        for name, store in self.stores.items():
            try:
                stats[name] = store.get_stats()
            except Exception as e:
                logger.error(f"Error getting stats for {name}: {e}")
                stats[name] = {}
        
        return stats
    
    def backup_collections(self, backup_path: str):
        """Backup all collections (placeholder for future implementation)."""
        logger.info(f"Backup functionality not yet implemented for path: {backup_path}")
        # This would involve exporting vectors and metadata
        # Implementation depends on specific backup requirements
    
    def list_stores(self) -> List[str]:
        """List all managed store names."""
        return list(self.stores.keys())
    
    def remove_store(self, name: str) -> bool:
        """Remove a store from management."""
        if name in self.stores:
            del self.stores[name]
            logger.info(f"Removed vector store: {name}")
            return True
        return False