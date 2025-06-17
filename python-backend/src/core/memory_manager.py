"""
Memory management system for FlutterSwarm agents.

This module provides memory management with text-based search
and context-aware retrieval.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from collections import defaultdict

from ..config import settings

from ..models import MemoryEntry
from ..config import get_logger, settings

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = get_logger("memory_manager")


class MemoryManager:
    """
    Memory management system with LLM integration.
    
    Features:
    - Short-term memory for session-based context
    - Long-term memory with text-based search
    - Context-aware retrieval for LLM prompting
    - Intelligent memory consolidation and cleanup
    """
    
    def __init__(self, agent_id: str, llm_client: Optional['LLMClient'] = None):
        self.agent_id = agent_id
        self.llm_client = llm_client
        self._short_term_memory: Dict[str, MemoryEntry] = {}
        self._long_term_memory: Dict[str, MemoryEntry] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._last_cleanup = datetime.utcnow()
        
        logger.info(f"MemoryManager initialized for agent {agent_id}")
        
    async def store_memory(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        importance: float = 1.0,
        long_term: bool = False
    ) -> str:
        """Store content in memory with optional metadata."""
        entry_id = f"{self.agent_id}_{datetime.utcnow().timestamp()}"
        
        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            metadata=metadata or {},
            importance=importance,
            agent_id=self.agent_id
        )
        
        # Store in appropriate memory layer
        if long_term:
            self._long_term_memory[entry_id] = entry
        else:
            self._short_term_memory[entry_id] = entry
        
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
        similarity_threshold: float = 0.7
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search memory using text matching."""
        return await self._text_search(query, limit)
    
    async def get_relevant_context(self, query: str, max_tokens: int = 2000) -> str:
        """Get relevant context for LLM prompting."""
        relevant_memories = await self.search_memory(query, limit=10)
        
        if not relevant_memories:
            return "No relevant context found in memory."
        
        context_parts = []
        current_length = 0
        
        for entry, similarity in relevant_memories:
            entry_tokens = len(entry.content) // 4  # Rough approximation
            
            if current_length + entry_tokens > max_tokens:
                break
            
            context_parts.append(f"[Relevance: {similarity:.2f}] {entry.content}")
            current_length += entry_tokens
        
        return "=== RELEVANT MEMORY CONTEXT ===\n\n" + "\n\n".join(context_parts)
    
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
                    max_tokens=settings.llm.max_tokens // 4,  # Use quarter of max tokens for consolidation
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
    
    async def _text_search(self, query: str, limit: int) -> List[Tuple[MemoryEntry, float]]:
        """Text-based search with simple scoring."""
        results = []
        query_words = set(query.lower().split())
        
        all_entries = {**self._short_term_memory, **self._long_term_memory}
        
        for entry in all_entries.values():
            content_words = set(entry.content.lower().split())
            
            # Calculate simple word overlap score
            common_words = query_words.intersection(content_words)
            if common_words:
                score = len(common_words) / len(query_words)
                # Boost score for exact phrase matches
                if query.lower() in entry.content.lower():
                    score += 0.5
                # Boost score based on importance
                score *= entry.importance
                results.append((entry, min(1.0, score)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _build_consolidation_prompt(self, entries: List[MemoryEntry]) -> str:
        """Build intelligent consolidation prompt for LLM."""
        content_list = []
        for i, entry in enumerate(entries, 1):
            content_list.append(f"{i}. [{entry.timestamp.isoformat()}] {entry.content}")
        
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
            contents.append(f"[{entry.timestamp.isoformat()}] {entry.content}")
        
        return f"CONSOLIDATED MEMORY:\n\n" + "\n\n".join(contents)
    
    def _merge_metadata(self, entries: List[MemoryEntry]) -> Dict[str, Any]:
        """Intelligently merge metadata from multiple entries."""
        merged = {}
        
        # Collect all metadata keys
        all_keys = set()
        for entry in entries:
            all_keys.update(entry.metadata.keys())
        
        for key in all_keys:
            values = [entry.metadata.get(key) for entry in entries if key in entry.metadata]
            if values:
                # Use most recent non-null value for single values
                merged[key] = values[-1]
        
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
                related.append(f"- {other_entry.content[:100]}...")
        
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
                if entry.importance > 0.7 or self._access_counts[entry_id] > 3:
                    # Promote to long-term
                    self._long_term_memory[entry_id] = entry
                    promoted_entries.append(entry_id)
                else:
                    expired_entries.append(entry_id)
                del self._short_term_memory[entry_id]
        
        # Clean up long-term memory if it gets too large
        max_entries = getattr(settings.agent, 'max_memory_entries', 1000)
        max_long_term = max_entries // 2
        
        if len(self._long_term_memory) > max_long_term:
            # Remove least accessed, oldest entries
            entries_by_access = sorted(
                self._long_term_memory.items(),
                key=lambda x: (self._access_counts[x[0]], x[1].timestamp)
            )
            
            to_remove = len(self._long_term_memory) - max_long_term
            for entry_id, _ in entries_by_access[:to_remove]:
                del self._long_term_memory[entry_id]
                if entry_id in self._access_counts:
                    del self._access_counts[entry_id]
        
        self._last_cleanup = current_time
        
        if expired_entries or promoted_entries:
            logger.info(f"Memory cleanup: expired {len(expired_entries)}, promoted {len(promoted_entries)}")
    
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
