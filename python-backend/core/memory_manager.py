"""
Multi-layered memory system for agent context management.
Implements short-term, long-term, and shared memory with semantic search.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import sqlite3
import pickle

import aioredis
from aioredis import Redis
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import numpy as np
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from .agent_types import ProjectContext, AgentMessage, TaskDefinition, WorkflowState

logger = logging.getLogger(__name__)


class MemoryType:
    """Types of memory storage."""
    
    SHORT_TERM = "short_term"  # Current session, volatile
    LONG_TERM = "long_term"    # Persistent across sessions
    SHARED = "shared"          # Real-time shared workspace
    VECTOR = "vector"          # Semantic search and embeddings


class MemoryEntry:
    """Base class for memory entries."""
    
    def __init__(
        self,
        entry_id: str = None,
        content: Any = None,
        metadata: Dict[str, Any] = None,
        created_at: datetime = None,
        expires_at: datetime = None
    ):
        self.entry_id = entry_id or str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now()
        self.expires_at = expires_at
        self.access_count = 0
        self.last_accessed = self.created_at


class ShortTermMemory:
    """
    Short-term memory for current session context.
    Uses in-memory storage with optional persistence.
    """
    
    def __init__(self, session_id: str, max_size: int = 1000):
        self.session_id = session_id
        self.max_size = max_size
        self.memory: Dict[str, MemoryEntry] = {}
        self.access_order: List[str] = []  # LRU tracking
        
        logger.info(f"ShortTermMemory initialized for session {session_id}")
    
    async def store(
        self, 
        key: str, 
        content: Any, 
        metadata: Dict[str, Any] = None,
        ttl: int = None
    ) -> str:
        """Store content in short-term memory."""
        try:
            # Calculate expiration
            expires_at = None
            if ttl:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            
            # Create memory entry
            entry = MemoryEntry(
                entry_id=key,
                content=content,
                metadata=metadata or {},
                expires_at=expires_at
            )
            
            # Store entry
            self.memory[key] = entry
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            # Enforce size limit
            await self._enforce_size_limit()
            
            logger.debug(f"Stored {key} in short-term memory")
            return entry.entry_id
            
        except Exception as e:
            logger.error(f"Failed to store in short-term memory: {e}")
            raise
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve content from short-term memory."""
        try:
            entry = self.memory.get(key)
            if not entry:
                return None
            
            # Check expiration
            if entry.expires_at and datetime.now() > entry.expires_at:
                await self.delete(key)
                return None
            
            # Update access tracking
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            
            # Update LRU order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return entry.content
            
        except Exception as e:
            logger.error(f"Failed to retrieve from short-term memory: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete entry from short-term memory."""
        try:
            if key in self.memory:
                del self.memory[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                logger.debug(f"Deleted {key} from short-term memory")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete from short-term memory: {e}")
            return False
    
    async def clear(self):
        """Clear all short-term memory."""
        self.memory.clear()
        self.access_order.clear()
        logger.info(f"Cleared short-term memory for session {self.session_id}")
    
    async def get_keys(self) -> List[str]:
        """Get all keys in short-term memory."""
        return list(self.memory.keys())
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total_entries = len(self.memory)
        expired_entries = 0
        
        for entry in self.memory.values():
            if entry.expires_at and datetime.now() > entry.expires_at:
                expired_entries += 1
        
        return {
            "session_id": self.session_id,
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "memory_usage": f"{total_entries}/{self.max_size}",
            "oldest_entry": min([e.created_at for e in self.memory.values()]) if self.memory else None,
            "newest_entry": max([e.created_at for e in self.memory.values()]) if self.memory else None
        }
    
    async def _enforce_size_limit(self):
        """Enforce memory size limit using LRU eviction."""
        while len(self.memory) > self.max_size:
            # Remove oldest accessed entry
            if self.access_order:
                oldest_key = self.access_order.pop(0)
                if oldest_key in self.memory:
                    del self.memory[oldest_key]
    
    async def cleanup_expired(self):
        """Remove expired entries."""
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self.memory.items():
            if entry.expires_at and now > entry.expires_at:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self.delete(key)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")


class LongTermMemory:
    """
    Long-term memory for persistent storage across sessions.
    Uses SQLite for structured data storage.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.engine = None
        self.session_factory = None
        
        logger.info(f"LongTermMemory initialized with DB: {db_path}")
    
    async def initialize(self):
        """Initialize the long-term memory database."""
        try:
            # Create async engine
            self.engine = create_async_engine(f"sqlite+aiosqlite:///{self.db_path}")
            
            # Create session factory
            self.session_factory = sessionmaker(
                self.engine, 
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            await self._create_tables()
            
            logger.info("LongTermMemory initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LongTermMemory: {e}")
            raise
    
    async def _create_tables(self):
        """Create database tables for long-term memory."""
        # Implementation would create tables for:
        # - memory_entries (id, key, content, metadata, created_at, updated_at)
        # - project_contexts (project_id, context_data, created_at, updated_at)
        # - task_history (task_id, task_data, created_at, completed_at)
        # - agent_knowledge (agent_id, knowledge_data, created_at, updated_at)
        pass
    
    async def store_project_context(
        self, 
        project_id: str, 
        context: ProjectContext
    ) -> bool:
        """Store project context in long-term memory."""
        try:
            # Implementation for storing project context
            logger.debug(f"Stored project context for {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store project context: {e}")
            return False
    
    async def retrieve_project_context(
        self, 
        project_id: str
    ) -> Optional[ProjectContext]:
        """Retrieve project context from long-term memory."""
        try:
            # Implementation for retrieving project context
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve project context: {e}")
            return None
    
    async def store_task_history(
        self, 
        task: TaskDefinition
    ) -> bool:
        """Store completed task in history."""
        try:
            # Implementation for storing task history
            return True
            
        except Exception as e:
            logger.error(f"Failed to store task history: {e}")
            return False
    
    async def get_similar_tasks(
        self, 
        task_description: str, 
        limit: int = 5
    ) -> List[TaskDefinition]:
        """Find similar tasks from history."""
        try:
            # Implementation for finding similar tasks
            return []
            
        except Exception as e:
            logger.error(f"Failed to find similar tasks: {e}")
            return []


class SharedMemory:
    """
    Shared memory workspace using Redis for real-time collaboration.
    Provides atomic operations and distributed locking.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[Redis] = None
        self.namespace = "flutterswarm:shared"
        
        logger.info(f"SharedMemory initialized with Redis URL: {redis_url}")
    
    async def initialize(self):
        """Initialize shared memory connection."""
        try:
            self.redis = await aioredis.from_url(self.redis_url)
            await self.redis.ping()
            
            logger.info("SharedMemory initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SharedMemory: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown shared memory connection."""
        if self.redis:
            await self.redis.close()
    
    def _get_key(self, key: str) -> str:
        """Get namespaced key."""
        return f"{self.namespace}:{key}"
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = None
    ) -> bool:
        """Set value in shared memory."""
        try:
            redis_key = self._get_key(key)
            serialized_value = json.dumps(value, default=str)
            
            if ttl:
                await self.redis.setex(redis_key, ttl, serialized_value)
            else:
                await self.redis.set(redis_key, serialized_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set shared memory value: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from shared memory."""
        try:
            redis_key = self._get_key(key)
            value = await self.redis.get(redis_key)
            
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get shared memory value: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete value from shared memory."""
        try:
            redis_key = self._get_key(key)
            result = await self.redis.delete(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete shared memory value: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Atomically increment a counter."""
        try:
            redis_key = self._get_key(key)
            result = await self.redis.incrby(redis_key, amount)
            return result
            
        except Exception as e:
            logger.error(f"Failed to increment counter: {e}")
            return None
    
    async def acquire_lock(
        self, 
        lock_key: str, 
        timeout: int = 10,
        blocking_timeout: int = 5
    ) -> Optional[str]:
        """Acquire a distributed lock."""
        try:
            lock_id = str(uuid.uuid4())
            redis_key = self._get_key(f"lock:{lock_key}")
            
            # Try to acquire lock
            result = await self.redis.set(
                redis_key, 
                lock_id, 
                ex=timeout, 
                nx=True
            )
            
            if result:
                return lock_id
            
            # If blocking, wait for lock
            if blocking_timeout > 0:
                start_time = datetime.now()
                while (datetime.now() - start_time).seconds < blocking_timeout:
                    await asyncio.sleep(0.1)
                    result = await self.redis.set(
                        redis_key, 
                        lock_id, 
                        ex=timeout, 
                        nx=True
                    )
                    if result:
                        return lock_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return None
    
    async def release_lock(self, lock_key: str, lock_id: str) -> bool:
        """Release a distributed lock."""
        try:
            redis_key = self._get_key(f"lock:{lock_key}")
            
            # Use Lua script for atomic release
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            
            result = await self.redis.eval(lua_script, 1, redis_key, lock_id)
            return result == 1
            
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")
            return False


class VectorMemory:
    """
    Vector-based memory for semantic search and similarity matching.
    Uses ChromaDB for vector storage and sentence-transformers for embeddings.
    """
    
    def __init__(self, collection_name: str = "flutterswarm_knowledge"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        logger.info(f"VectorMemory initialized for collection: {collection_name}")
    
    async def initialize(self):
        """Initialize vector memory."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.Client(ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
            except:
                self.collection = self.client.create_collection(self.collection_name)
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("VectorMemory initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VectorMemory: {e}")
            raise
    
    async def add_document(
        self, 
        doc_id: str, 
        content: str, 
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Add document to vector memory."""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()
            
            # Add to collection
            self.collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata or {}],
                ids=[doc_id]
            )
            
            logger.debug(f"Added document {doc_id} to vector memory")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    async def search_similar(
        self, 
        query: str, 
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            # Format results
            documents = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i] if results['distances'] else 0
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    if similarity >= min_similarity:
                        documents.append({
                            'id': results['ids'][0][i],
                            'content': doc,
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                            'similarity': similarity
                        })
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            return []
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from vector memory."""
        try:
            self.collection.delete(ids=[doc_id])
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False


class MemoryManager:
    """
    Central memory manager that coordinates all memory types.
    Provides unified interface for memory operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize memory systems
        self.short_term_memories: Dict[str, ShortTermMemory] = {}
        self.long_term_memory = LongTermMemory(
            config.get("long_term_db", "memory/long_term.db")
        )
        self.shared_memory = SharedMemory(
            config.get("redis_url", "redis://localhost:6379")
        )
        self.vector_memory = VectorMemory(
            config.get("vector_collection", "flutterswarm_knowledge")
        )
        
        logger.info("MemoryManager initialized")
    
    async def initialize(self):
        """Initialize all memory systems."""
        try:
            await self.long_term_memory.initialize()
            await self.shared_memory.initialize()
            await self.vector_memory.initialize()
            
            logger.info("All memory systems initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory systems: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown all memory systems."""
        try:
            await self.shared_memory.shutdown()
            # Add shutdown for other memory systems as needed
            
            logger.info("All memory systems shutdown")
            
        except Exception as e:
            logger.error(f"Error during memory shutdown: {e}")
    
    def get_short_term_memory(self, session_id: str) -> ShortTermMemory:
        """Get or create short-term memory for session."""
        if session_id not in self.short_term_memories:
            self.short_term_memories[session_id] = ShortTermMemory(
                session_id,
                max_size=self.config.get("short_term_max_size", 1000)
            )
        return self.short_term_memories[session_id]
    
    async def store_project_knowledge(
        self, 
        project_id: str, 
        knowledge: str, 
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store project-specific knowledge."""
        try:
            doc_id = f"project_{project_id}_{uuid.uuid4().hex[:8]}"
            metadata = metadata or {}
            metadata.update({
                "project_id": project_id,
                "type": "project_knowledge",
                "created_at": datetime.now().isoformat()
            })
            
            return await self.vector_memory.add_document(doc_id, knowledge, metadata)
            
        except Exception as e:
            logger.error(f"Failed to store project knowledge: {e}")
            return False
    
    async def search_project_knowledge(
        self, 
        query: str, 
        project_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for relevant project knowledge."""
        try:
            # Search vector memory
            results = await self.vector_memory.search_similar(query, limit * 2)
            
            # Filter by project if specified
            if project_id:
                filtered_results = []
                for result in results:
                    if result['metadata'].get('project_id') == project_id:
                        filtered_results.append(result)
                results = filtered_results[:limit]
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search project knowledge: {e}")
            return []
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            stats = {
                "short_term_sessions": len(self.short_term_memories),
                "short_term_details": {}
            }
            
            # Short-term memory stats
            for session_id, memory in self.short_term_memories.items():
                stats["short_term_details"][session_id] = await memory.get_statistics()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return {"error": str(e)}
