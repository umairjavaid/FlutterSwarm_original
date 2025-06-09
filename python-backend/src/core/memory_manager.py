"""
Enhanced Memory management system for FlutterSwarm agents.

This module provides dual-layer memory management with LLM-powered embeddings
for semantic search and context-aware retrieval.
"""
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from collections import defaultdict
import numpy as np

from ..models import MemoryEntry
from ..config import get_logger, settings

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = get_logger("memory_manager")


class MemoryManager:
    """
    Enhanced dual-layer memory management system with LLM integration.
    
    Features:
    - Short-term memory for session-based context
    - Long-term memory with vector embeddings for semantic search  
    - LLM-powered embedding generation with fallback
    - Context-aware retrieval for LLM prompting
    - Intelligent memory consolidation and cleanup
    - Access pattern tracking and optimization
    """
    
    def __init__(self, agent_id: str, llm_client: Optional['LLMClient'] = None):
        self.agent_id = agent_id
        self.llm_client = llm_client
        self._short_term_memory: Dict[str, MemoryEntry] = {}
        self._long_term_memory: Dict[str, MemoryEntry] = {}
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._last_cleanup = datetime.utcnow()
        
        # OpenAI ada-002 embedding dimension
        self._embedding_dim = 1536
        
        logger.info(f"MemoryManager initialized for agent {agent_id}")
        
    async def store_memory(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        importance: float = 1.0,
        long_term: bool = False
    ) -> str:
        """
        Store content in memory with optional metadata and embedding generation.
        
        Args:
            content: Content to store
            metadata: Optional metadata dictionary
            correlation_id: Optional correlation ID for workflow tracking
            importance: Importance score (0-1)
            long_term: Whether to store in long-term memory
            
        Returns:
            Memory entry ID
        """
        entry_id = f"{self.agent_id}_{datetime.utcnow().timestamp()}"
        
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata or {},
            correlation_id=correlation_id,
            importance=importance,
            agent_id=self.agent_id
        )
        
        # Generate embedding for semantic search
        try:
            entry.embedding = await self._generate_embedding(content)
        except Exception as e:
            logger.warning(f"Failed to generate embedding for entry {entry_id}: {e}")
            entry.embedding = None
        
        # Store in appropriate memory layer
        if long_term:
            self._long_term_memory[entry_id] = entry
            logger.debug(f"Stored in long-term memory: {entry_id}")
        else:
            self._short_term_memory[entry_id] = entry
            logger.debug(f"Stored in short-term memory: {entry_id}")
        
        # Trigger cleanup if memory is getting full
        total_entries = len(self._short_term_memory) + len(self._long_term_memory)
        max_entries = getattr(settings.agent, 'max_memory_entries', 1000)
        
        if total_entries > max_entries:
            await self._cleanup_memory()
        
        return entry_id
    
    async def retrieve_memory(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a specific memory entry and update access tracking.
        
        Args:
            entry_id: ID of the memory entry
            
        Returns:
            Memory entry if found, None otherwise
        """
        # Check short-term memory first (faster access)
        if entry_id in self._short_term_memory:
            entry = self._short_term_memory[entry_id]
            entry.update_access()
            self._access_counts[entry_id] += 1
            return entry
        
        # Check long-term memory
        if entry_id in self._long_term_memory:
            entry = self._long_term_memory[entry_id]
            entry.update_access()
            self._access_counts[entry_id] += 1
            return entry
        
        logger.debug(f"Memory entry not found: {entry_id}")
        return None
    
    async def search_memory(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        search_metadata: bool = True,
        boost_recent: bool = True
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Search memory using semantic similarity with advanced ranking.
        
        Args:
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            search_metadata: Whether to include metadata in search
            boost_recent: Whether to boost recent memories in ranking
            
        Returns:
            List of (memory_entry, similarity_score) tuples sorted by relevance
        """
        try:
            query_embedding = await self._generate_embedding(query)
            if query_embedding is None:
                logger.warning("Failed to generate query embedding, falling back to text search")
                return await self._text_search(query, limit)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return await self._text_search(query, limit)
        
        results = []
        current_time = datetime.utcnow()
        
        # Search both memory layers
        all_entries = {**self._short_term_memory, **self._long_term_memory}
        
        for entry in all_entries.values():
            if entry.embedding is None:
                continue
            
            # Calculate semantic similarity
            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            
            if similarity >= similarity_threshold:
                # Apply ranking boosts
                final_score = similarity
                
                # Boost for importance
                final_score *= (0.5 + 0.5 * entry.importance)
                
                # Boost for recent entries if enabled
                if boost_recent:
                    age_hours = (current_time - entry.timestamp).total_seconds() / 3600
                    recency_factor = max(0.5, 1.0 - (age_hours / 168))  # Decay over 1 week
                    final_score *= recency_factor
                
                # Boost for frequently accessed entries
                access_boost = min(1.2, 1.0 + (self._access_counts[entry.id] * 0.02))
                final_score *= access_boost
                
                results.append((entry, final_score))
                entry.update_access()
                self._access_counts[entry.id] += 1
        
        # Sort by final score and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    async def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 2000,
        include_correlations: bool = True,
        context_format: str = "structured"
    ) -> str:
        """
        Get relevant context for LLM prompting with intelligent formatting.
        
        Args:
            query: Query to find relevant context for
            max_tokens: Maximum tokens for context (approximate)
            include_correlations: Whether to include correlated memories
            context_format: Format style ("structured", "narrative", "bullet")
            
        Returns:
            Formatted context string optimized for LLM consumption
        """
        # Search for relevant memories
        relevant_memories = await self.search_memory(query, limit=20, similarity_threshold=0.6)
        
        if not relevant_memories:
            return "No relevant context found in memory."
        
        context_parts = []
        current_length = 0
        
        for entry, similarity in relevant_memories:
            # Estimate token count (rough approximation: 1 token ≈ 4 characters)
            entry_tokens = len(entry.content) // 4
            
            if current_length + entry_tokens > max_tokens:
                break
            
            # Format context based on style
            if context_format == "structured":
                context_part = f"[Relevance: {similarity:.2f}] {entry.content}"
                if entry.metadata:
                    key_metadata = {k: v for k, v in entry.metadata.items() 
                                  if k in ['type', 'category', 'source', 'priority']}
                    if key_metadata:
                        context_part += f" [Meta: {json.dumps(key_metadata)}]"
            elif context_format == "narrative":
                context_part = entry.content
            else:  # bullet format
                context_part = f"• {entry.content}"
            
            context_parts.append(context_part)
            current_length += entry_tokens
        
        # Format final context
        if context_format == "structured":
            context = "=== RELEVANT MEMORY CONTEXT ===\n\n" + "\n\n".join(context_parts)
        elif context_format == "narrative":
            context = "Based on previous context:\n\n" + "\n\n".join(context_parts)
        else:  # bullet
            context = "Relevant information:\n" + "\n".join(context_parts)
        
        # Add correlation information if requested
        if include_correlations and relevant_memories:
            correlations = await self._get_correlations(relevant_memories[0][0])
            if correlations:
                context += f"\n\n=== RELATED CONTEXT ===\n{correlations}"
        
        return context
    
    async def consolidate_memories(self, correlation_id: str) -> Optional[str]:
        """
        Consolidate related memories by correlation ID using LLM reasoning.
        
        Args:
            correlation_id: Correlation ID to consolidate
            
        Returns:
            ID of consolidated memory entry if successful
        """
        related_entries = []
        
        # Find all entries with the correlation ID
        for entry in {**self._short_term_memory, **self._long_term_memory}.values():
            if entry.correlation_id == correlation_id:
                related_entries.append(entry)
        
        if len(related_entries) < 2:
            logger.debug(f"Not enough entries to consolidate for correlation_id: {correlation_id}")
            return None
        
        # Sort entries by timestamp for chronological consolidation
        related_entries.sort(key=lambda x: x.timestamp)
        
        # Use LLM to create intelligent consolidation if available
        if self.llm_client:
            try:
                consolidation_prompt = self._build_consolidation_prompt(related_entries)
                consolidated_content = await self.llm_client.generate(
                    prompt=consolidation_prompt,
                    temperature=0.3,
                    max_tokens=1000,
                    agent_id=self.agent_id,
                    correlation_id=correlation_id
                )
            except Exception as e:
                logger.warning(f"LLM consolidation failed, using simple merge: {e}")
                consolidated_content = self._simple_consolidation(related_entries)
        else:
            consolidated_content = self._simple_consolidation(related_entries)
        
        # Merge metadata intelligently
        consolidated_metadata = self._merge_metadata(related_entries)
        
        # Calculate weighted importance
        total_importance = sum(entry.importance for entry in related_entries)
        avg_importance = total_importance / len(related_entries)
        
        # Store consolidated memory in long-term
        consolidated_id = await self.store_memory(
            content=consolidated_content,
            metadata=consolidated_metadata,
            correlation_id=correlation_id,
            importance=min(1.0, avg_importance * 1.1),  # Slight boost for consolidated entries
            long_term=True
        )
        
        # Remove original entries from short-term memory (keep in long-term for now)
        removed_count = 0
        for entry in related_entries:
            if entry.id in self._short_term_memory:
                del self._short_term_memory[entry.id]
                removed_count += 1
        
        logger.info(f"Consolidated {len(related_entries)} memories into {consolidated_id} (removed {removed_count} from short-term)")
        return consolidated_id
    
    async def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text using LLM client with intelligent fallback.
        """
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._embeddings_cache:
            return self._embeddings_cache[text_hash]
        
        try:
            # Use LLM client if available
            if self.llm_client:
                try:
                    embedding_list = await self.llm_client.generate_embedding(text)
                    embedding = np.array(embedding_list, dtype=np.float32)
                    
                    # Cache the embedding
                    self._embeddings_cache[text_hash] = embedding
                    return embedding
                    
                except Exception as e:
                    logger.warning(f"LLM embedding failed, using fallback: {e}")
            
            # Fallback to deterministic hash-based approach
            embedding = self._generate_fallback_embedding(text)
            
            # Cache the fallback embedding
            self._embeddings_cache[text_hash] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """Generate deterministic embedding using multiple hash functions."""
        # Use multiple hash functions for better distribution
        hash_functions = [
            lambda x: hashlib.md5(x.encode()).digest(),
            lambda x: hashlib.sha1(x.encode()).digest(),
            lambda x: hashlib.sha256(x.encode()).digest(),
            lambda x: hashlib.blake2b(x.encode(), digest_size=32).digest()
        ]
        
        # Combine hash outputs
        combined_hash = b''.join(hash_func(text) for hash_func in hash_functions)
        
        # Convert to float array
        embedding = np.frombuffer(combined_hash, dtype=np.uint8)[:self._embedding_dim]
        
        # Pad if necessary
        if len(embedding) < self._embedding_dim:
            padding = np.zeros(self._embedding_dim - len(embedding), dtype=np.uint8)
            embedding = np.concatenate([embedding, padding])
        
        # Normalize
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity with error handling."""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return float(dot_product / (norm_a * norm_b))
        except Exception:
            return 0.0
    
    async def _text_search(self, query: str, limit: int) -> List[Tuple[MemoryEntry, float]]:
        """Advanced text-based search with TF-IDF-like scoring."""
        results = []
        query_words = set(query.lower().split())
        
        all_entries = {**self._short_term_memory, **self._long_term_memory}
        
        for entry in all_entries.values():
            content_words = set(entry.content.lower().split())
            
            # Calculate Jaccard similarity
            intersection = query_words.intersection(content_words)
            union = query_words.union(content_words)
            
            if intersection:
                jaccard_score = len(intersection) / len(union)
                
                # Boost for exact phrase matches
                phrase_boost = 1.0
                if query.lower() in entry.content.lower():
                    phrase_boost = 1.5
                
                # Boost for metadata matches
                metadata_boost = 1.0
                if entry.metadata:
                    metadata_text = ' '.join(str(v) for v in entry.metadata.values()).lower()
                    if any(word in metadata_text for word in query_words):
                        metadata_boost = 1.2
                
                final_score = jaccard_score * phrase_boost * metadata_boost
                results.append((entry, final_score))
                entry.update_access()
                self._access_counts[entry.id] += 1
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _build_consolidation_prompt(self, entries: List[MemoryEntry]) -> str:
        """Build intelligent consolidation prompt for LLM."""
        content_list = []
        for i, entry in enumerate(entries, 1):
            timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            content_list.append(f"{i}. [{timestamp}] {entry.content}")
        
        return f"""
Please consolidate the following related memory entries into a single, coherent summary:

{chr(10).join(content_list)}

Requirements:
- Preserve all important information
- Maintain chronological context where relevant
- Eliminate redundancy while keeping key details
- Create a clear, well-structured summary
- Focus on actionable insights and outcomes

Consolidated Summary:"""
    
    def _simple_consolidation(self, entries: List[MemoryEntry]) -> str:
        """Simple consolidation fallback method."""
        contents = []
        for entry in entries:
            timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M")
            contents.append(f"[{timestamp}] {entry.content}")
        
        return f"CONSOLIDATED MEMORY:\n\n" + "\n\n".join(contents)
    
    def _merge_metadata(self, entries: List[MemoryEntry]) -> Dict[str, Any]:
        """Intelligently merge metadata from multiple entries."""
        merged = {}
        
        # Collect all metadata keys
        all_keys = set()
        for entry in entries:
            all_keys.update(entry.metadata.keys())
        
        for key in all_keys:
            values = []
            for entry in entries:
                if key in entry.metadata:
                    value = entry.metadata[key]
                    if value not in values:
                        values.append(value)
            
            # Store as list if multiple values, single value otherwise
            if len(values) == 1:
                merged[key] = values[0]
            elif len(values) > 1:
                merged[key] = values
        
        # Add consolidation metadata
        merged['consolidated'] = True
        merged['source_count'] = len(entries)
        merged['consolidation_timestamp'] = datetime.utcnow().isoformat()
        
        return merged
    
    async def _get_correlations(self, entry: MemoryEntry) -> str:
        """Get formatted correlated memories for an entry."""
        if not entry.correlation_id:
            return ""
        
        related = []
        all_entries = {**self._short_term_memory, **self._long_term_memory}
        
        for other_entry in all_entries.values():
            if (other_entry.correlation_id == entry.correlation_id and 
                other_entry.id != entry.id):
                # Format with timestamp and truncate content
                timestamp = other_entry.timestamp.strftime("%m/%d %H:%M")
                content = other_entry.content[:100] + "..." if len(other_entry.content) > 100 else other_entry.content
                related.append(f"[{timestamp}] {content}")
        
        return "\n".join(related) if related else ""
    
    async def _cleanup_memory(self) -> None:
        """Intelligent memory cleanup with access pattern analysis."""
        current_time = datetime.utcnow()
        
        # Only cleanup if enough time has passed (avoid thrashing)
        if current_time - self._last_cleanup < timedelta(minutes=5):
            return
        
        logger.debug("Starting memory cleanup")
        
        # Clean up short-term memory based on TTL and importance
        memory_ttl = getattr(settings.agent, 'memory_ttl', 3600)  # Default 1 hour
        ttl_cutoff = current_time - timedelta(seconds=memory_ttl)
        
        expired_entries = []
        promoted_entries = []
        
        for entry_id, entry in list(self._short_term_memory.items()):
            if entry.timestamp < ttl_cutoff:
                # Decide whether to promote to long-term or delete
                access_count = self._access_counts.get(entry_id, 0)
                
                # Promotion criteria: high importance OR frequent access OR has correlations
                should_promote = (
                    entry.importance > 0.7 or 
                    access_count > 3 or
                    entry.correlation_id is not None
                )
                
                if should_promote:
                    self._long_term_memory[entry_id] = entry
                    promoted_entries.append(entry_id)
                else:
                    expired_entries.append(entry_id)
                
                del self._short_term_memory[entry_id]
        
        # Clean up long-term memory if it gets too large
        max_entries = getattr(settings.agent, 'max_memory_entries', 1000)
        max_long_term = max_entries // 2
        
        if len(self._long_term_memory) > max_long_term:
            # Calculate composite score for each entry
            scored_entries = []
            for entry_id, entry in self._long_term_memory.items():
                access_count = self._access_counts.get(entry_id, 0)
                age_days = (current_time - entry.timestamp).days
                
                # Composite score: importance + access frequency - age penalty
                score = (
                    entry.importance * 0.4 +
                    min(access_count / 10, 0.3) +  # Cap access contribution
                    (0.3 if entry.correlation_id else 0) -  # Correlation bonus
                    (age_days * 0.01)  # Age penalty
                )
                
                scored_entries.append((entry_id, score))
            
            # Sort by score and remove lowest scoring entries
            scored_entries.sort(key=lambda x: x[1])
            remove_count = len(scored_entries) - max_long_term
            
            removed_entries = []
            for entry_id, _ in scored_entries[:remove_count]:
                del self._long_term_memory[entry_id]
                if entry_id in self._access_counts:
                    del self._access_counts[entry_id]
                removed_entries.append(entry_id)
            
            logger.debug(f"Removed {len(removed_entries)} low-scoring long-term memories")
        
        # Clean up embedding cache if it gets too large
        if len(self._embeddings_cache) > 1000:
            # Remove oldest 20% of cached embeddings
            cache_items = list(self._embeddings_cache.items())
            remove_count = len(cache_items) // 5
            for i in range(remove_count):
                del self._embeddings_cache[cache_items[i][0]]
        
        self._last_cleanup = current_time
        
        if expired_entries or promoted_entries:
            logger.info(f"Memory cleanup completed: {len(expired_entries)} expired, {len(promoted_entries)} promoted to long-term")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        current_time = datetime.utcnow()
        
        # Calculate average access counts
        access_values = list(self._access_counts.values())
        avg_access = sum(access_values) / len(access_values) if access_values else 0
        
        # Calculate age statistics
        all_entries = {**self._short_term_memory, **self._long_term_memory}
        if all_entries:
            ages = [(current_time - entry.timestamp).total_seconds() / 3600 for entry in all_entries.values()]
            avg_age_hours = sum(ages) / len(ages)
            oldest_hours = max(ages)
        else:
            avg_age_hours = 0
            oldest_hours = 0
        
        return {
            "agent_id": self.agent_id,
            "short_term_count": len(self._short_term_memory),
            "long_term_count": len(self._long_term_memory),
            "total_memories": len(all_entries),
            "embeddings_cached": len(self._embeddings_cache),
            "total_accesses": sum(self._access_counts.values()),
            "average_accesses": avg_access,
            "average_age_hours": avg_age_hours,
            "oldest_memory_hours": oldest_hours,
            "last_cleanup": self._last_cleanup.isoformat(),
            "llm_client_available": self.llm_client is not None
        }
    
    async def export_memories(self, correlation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export memories for backup or analysis."""
        all_entries = {**self._short_term_memory, **self._long_term_memory}
        
        if correlation_id:
            entries = [entry for entry in all_entries.values() 
                      if entry.correlation_id == correlation_id]
        else:
            entries = list(all_entries.values())
        
        return [entry.to_dict() for entry in entries]
    
    async def import_memories(self, memories_data: List[Dict[str, Any]]) -> int:
        """Import memories from backup data."""
        imported_count = 0
        
        for memory_data in memories_data:
            try:
                entry = MemoryEntry.from_dict(memory_data)
                
                # Store in appropriate memory layer based on age and importance
                age_hours = (datetime.utcnow() - entry.timestamp).total_seconds() / 3600
                is_long_term = age_hours > 24 or entry.importance > 0.7
                
                if is_long_term:
                    self._long_term_memory[entry.id] = entry
                else:
                    self._short_term_memory[entry.id] = entry
                
                imported_count += 1
                
            except Exception as e:
                logger.error(f"Failed to import memory: {e}")
                continue
        
        logger.info(f"Imported {imported_count} memories")
        return imported_count
