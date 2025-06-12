"""
LangGraph Checkpointer Implementation for FlutterSwarm.

This module provides persistent state management for LangGraph workflows
using Redis as the backend storage.
"""
import redis.asyncio as redis
import json
import pickle
from typing import Any, Dict, Optional, List, Tuple, AsyncIterator
from datetime import datetime, timedelta
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.memory import MemorySaver
from ..config import get_logger

logger = get_logger("langgraph_checkpointer")


class RedisCheckpointSaver(BaseCheckpointSaver):
    """
    Redis-based checkpoint saver for LangGraph state persistence.
    
    This implementation provides durable state storage across sessions
    and supports checkpoint versioning and cleanup.
    """
    
    def __init__(
        self,
        redis_url: str = None,
        key_prefix: str = "flutterswarm:checkpoint:",
        ttl_seconds: int = None,
        max_versions: int = None
    ):
        """
        Initialize Redis checkpoint saver.
        
        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for Redis keys
            ttl_seconds: Time-to-live for checkpoints
            max_versions: Maximum versions to keep per thread
        """
        from ..config import settings
        
        super().__init__()
        # Use config values if not explicitly provided
        self.redis_url = redis_url or settings.redis.url
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds or settings.redis.ttl_seconds
        self.max_versions = max_versions or settings.redis.max_versions
        self._redis: Optional[redis.Redis] = None
    
    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection, creating if necessary."""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=False)
        return self._redis
    
    def _make_key(self, thread_id: str, checkpoint_id: Optional[str] = None) -> str:
        """Create Redis key for checkpoint."""
        if checkpoint_id:
            return f"{self.key_prefix}{thread_id}:{checkpoint_id}"
        return f"{self.key_prefix}{thread_id}:*"
    
    async def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any]
    ) -> RunnableConfig:
        """Save checkpoint to Redis."""
        try:
            redis_client = await self._get_redis()
            thread_id = config["configurable"]["thread_id"]
            checkpoint_id = checkpoint["id"]
            
            # Serialize checkpoint data
            checkpoint_data = {
                "checkpoint": pickle.dumps(checkpoint),
                "metadata": json.dumps(metadata),
                "new_versions": json.dumps(new_versions),
                "timestamp": datetime.utcnow().timestamp()
            }
            
            # Store in Redis
            key = self._make_key(thread_id, checkpoint_id)
            await redis_client.hset(key, mapping=checkpoint_data)
            await redis_client.expire(key, self.ttl_seconds)
            
            # Cleanup old versions
            await self._cleanup_old_versions(redis_client, thread_id)
            
            logger.debug(f"Saved checkpoint {checkpoint_id} for thread {thread_id}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    async def get(
        self,
        config: RunnableConfig
    ) -> Optional[Checkpoint]:
        """Retrieve checkpoint from Redis."""
        try:
            redis_client = await self._get_redis()
            thread_id = config["configurable"]["thread_id"]
            
            # Get latest checkpoint
            latest_checkpoint = await self._get_latest_checkpoint(redis_client, thread_id)
            return latest_checkpoint
            
        except Exception as e:
            logger.error(f"Failed to get checkpoint: {e}")
            return None
    
    async def list(
        self,
        config: RunnableConfig,
        limit: Optional[int] = None,
        before: Optional[CheckpointMetadata] = None
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints for a thread."""
        try:
            redis_client = await self._get_redis()
            thread_id = config["configurable"]["thread_id"]
            
            pattern = self._make_key(thread_id)
            keys = await redis_client.keys(pattern)
            
            # Sort keys by timestamp
            key_timestamps = []
            for key in keys:
                data = await redis_client.hgetall(key)
                if b'timestamp' in data:
                    timestamp = float(data[b'timestamp'])
                    key_timestamps.append((key, timestamp))
            
            # Sort by timestamp (newest first)
            key_timestamps.sort(key=lambda x: x[1], reverse=True)
            
            # Apply limit
            if limit:
                key_timestamps = key_timestamps[:limit]
            
            # Yield checkpoints
            for key, timestamp in key_timestamps:
                data = await redis_client.hgetall(key)
                if b'checkpoint' in data and b'metadata' in data:
                    checkpoint = pickle.loads(data[b'checkpoint'])
                    metadata = json.loads(data[b'metadata'])
                    
                    yield CheckpointTuple(
                        config=config,
                        checkpoint=checkpoint,
                        metadata=metadata
                    )
                
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
    
    async def _get_latest_checkpoint(
        self,
        redis_client: redis.Redis,
        thread_id: str
    ) -> Optional[Checkpoint]:
        """Get the most recent checkpoint for a thread."""
        pattern = self._make_key(thread_id)
        keys = await redis_client.keys(pattern)
        
        if not keys:
            return None
        
        latest_key = None
        latest_timestamp = None
        
        for key in keys:
            data = await redis_client.hgetall(key)
            if b'timestamp' in data:
                timestamp = float(data[b'timestamp'])
                if latest_timestamp is None or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_key = key
        
        if latest_key:
            data = await redis_client.hgetall(latest_key)
            if b'checkpoint' in data:
                return pickle.loads(data[b'checkpoint'])
        
        return None
    
    async def _cleanup_old_versions(
        self,
        redis_client: redis.Redis,
        thread_id: str
    ) -> None:
        """Remove old checkpoint versions beyond the limit."""
        pattern = self._make_key(thread_id)
        keys = await redis_client.keys(pattern)
        
        if len(keys) <= self.max_versions:
            return
        
        # Get keys with timestamps
        key_timestamps = []
        for key in keys:
            data = await redis_client.hgetall(key)
            if b'timestamp' in data:
                timestamp = float(data[b'timestamp'])
                key_timestamps.append((key, timestamp))
        
        # Sort by timestamp (oldest first)
        key_timestamps.sort(key=lambda x: x[1])
        
        # Remove oldest versions
        keys_to_remove = key_timestamps[:-self.max_versions]
        for key, _ in keys_to_remove:
            await redis_client.delete(key)
            logger.debug(f"Removed old checkpoint: {key}")
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()


class MemoryCheckpointSaver(MemorySaver):
    """
    Extended memory-based checkpoint saver for development.
    
    This is useful for testing and development when Redis is not available.
    """
    
    def __init__(self, max_checkpoints: int = 100):
        """Initialize with checkpoint limit."""
        super().__init__()
        self.max_checkpoints = max_checkpoints

    async def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any]
    ) -> RunnableConfig:
        result = super().put(config, checkpoint, metadata, new_versions)
        await self._cleanup_old_checkpoints()
        return result
    
    async def _cleanup_old_checkpoints(self) -> None:
        """Cleanup old checkpoints when limit is exceeded."""
        if len(self.storage) > self.max_checkpoints:
            # Remove oldest checkpoints
            sorted_keys = sorted(self.storage.keys())
            keys_to_remove = sorted_keys[:-self.max_checkpoints]
            for key in keys_to_remove:
                del self.storage[key]


def create_checkpointer(
    backend: str = "memory",
    redis_url: Optional[str] = None,
    **kwargs
) -> BaseCheckpointSaver:
    """
    Create a checkpointer instance based on the specified backend.
    
    Args:
        backend: Type of checkpointer ("memory" or "redis")
        redis_url: Redis connection URL (required for redis backend)
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured checkpointer instance
    
    Raises:
        ValueError: If backend type is unsupported or configuration is invalid
    """
    if backend == "memory":
        return MemoryCheckpointSaver(
            max_checkpoints=kwargs.get("max_checkpoints", 100)
        )
    
    elif backend == "redis":
        from ..config import settings
        if not redis_url:
            redis_url = settings.redis.url
        
        return RedisCheckpointSaver(
            redis_url=redis_url,
            key_prefix=kwargs.get("key_prefix", "flutterswarm:checkpoint:"),
            ttl_seconds=kwargs.get("ttl_seconds", 86400 * 7),
            max_versions=kwargs.get("max_versions", 10)
        )
    
    else:
        raise ValueError(f"Unsupported checkpointer backend: {backend}")


async def cleanup_checkpoints(
    checkpointer: BaseCheckpointSaver,
    older_than_days: int = 7
) -> int:
    """
    Clean up old checkpoints from the storage backend.
    
    Args:
        checkpointer: Checkpointer instance to clean up
        older_than_days: Remove checkpoints older than this many days
    
    Returns:
        Number of checkpoints removed
    """
    cleanup_count = 0
    
    try:
        if isinstance(checkpointer, RedisCheckpointSaver):
            redis_client = await checkpointer._get_redis()
            pattern = f"{checkpointer.key_prefix}*"
            keys = await redis_client.keys(pattern)
            
            cutoff_time = (datetime.utcnow() - timedelta(days=older_than_days)).timestamp()
            
            for key in keys:
                data = await redis_client.hgetall(key)
                if b'timestamp' in data:
                    timestamp = float(data[b'timestamp'])
                    if timestamp < cutoff_time:
                        await redis_client.delete(key)
                        cleanup_count += 1
        
        logger.info(f"Cleaned up {cleanup_count} old checkpoints")
        
    except Exception as e:
        logger.error(f"Failed to cleanup checkpoints: {e}")
    
    return cleanup_count
