"""
Event-driven message broker for agent communication.
Implements publish-subscribe pattern with Redis backend.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import aioredis
from aioredis import Redis
from pydantic import BaseModel

from .agent_types import AgentMessage, MessageType, AgentType, Priority

logger = logging.getLogger(__name__)


@dataclass
class MessageHandler:
    """Handler for processing messages."""
    
    handler_id: str
    agent_id: str
    message_types: Set[MessageType]
    callback: Callable
    filter_func: Optional[Callable] = None


class MessageBroker:
    """
    Event-driven message broker for agent communication.
    Provides publish-subscribe messaging with Redis backend.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[Redis] = None
        self.pubsub = None
        
        # Message handlers
        self.handlers: Dict[str, MessageHandler] = {}
        self.topic_handlers: Dict[str, List[str]] = {}  # topic -> handler_ids
        
        # Message tracking
        self.message_history: List[AgentMessage] = []
        self.pending_responses: Dict[str, asyncio.Future] = {}
        
        # Configuration
        self.message_ttl = 3600  # 1 hour
        self.max_history_size = 10000
        
        # Event loop management
        self.running = False
        self.message_processor_task: Optional[asyncio.Task] = None
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"MessageBroker initialized with Redis URL: {redis_url}")
    
    async def initialize(self):
        """Initialize the message broker."""
        try:
            # Connect to Redis
            self.redis = await aioredis.from_url(self.redis_url)
            self.pubsub = self.redis.pubsub()
            
            # Test connection
            await self.redis.ping()
            
            # Start message processor
            self.running = True
            self.message_processor_task = asyncio.create_task(self._process_messages())
            
            logger.info("MessageBroker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MessageBroker: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the message broker."""
        try:
            self.running = False
            
            # Cancel message processor
            if self.message_processor_task:
                self.message_processor_task.cancel()
                try:
                    await self.message_processor_task
                except asyncio.CancelledError:
                    pass
            
            # Close Redis connections
            if self.pubsub:
                await self.pubsub.close()
            
            if self.redis:
                await self.redis.close()
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            logger.info("MessageBroker shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during MessageBroker shutdown: {e}")
    
    async def publish(
        self, 
        message: AgentMessage, 
        topic: Optional[str] = None
    ) -> bool:
        """
        Publish a message to the specified topic.
        
        Args:
            message: The message to publish
            topic: Optional topic override (defaults to message type)
            
        Returns:
            True if message was published successfully
        """
        try:
            if not self.redis:
                raise RuntimeError("MessageBroker not initialized")
            
            # Use message type as topic if not specified
            if not topic:
                topic = message.message_type.value
            
            # Serialize message
            message_data = message.model_dump_json()
            
            # Publish to Redis
            await self.redis.publish(topic, message_data)
            
            # Store in history
            self._add_to_history(message)
            
            # Handle response tracking
            if message.requires_response:
                self._track_response_required(message)
            
            logger.debug(f"Published message {message.message_id} to topic {topic}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    async def subscribe(
        self, 
        handler: MessageHandler, 
        topics: Optional[List[str]] = None
    ):
        """
        Subscribe a handler to specific topics.
        
        Args:
            handler: The message handler to register
            topics: List of topics to subscribe to (defaults to handler's message types)
        """
        try:
            if not self.redis:
                raise RuntimeError("MessageBroker not initialized")
            
            # Register handler
            self.handlers[handler.handler_id] = handler
            
            # Determine topics to subscribe to
            if not topics:
                topics = [msg_type.value for msg_type in handler.message_types]
            
            # Subscribe to topics
            for topic in topics:
                await self.pubsub.subscribe(topic)
                
                # Track topic handlers
                if topic not in self.topic_handlers:
                    self.topic_handlers[topic] = []
                self.topic_handlers[topic].append(handler.handler_id)
            
            logger.info(f"Subscribed handler {handler.handler_id} to topics: {topics}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe handler: {e}")
            raise
    
    async def unsubscribe(self, handler_id: str):
        """Unsubscribe a handler from all topics."""
        try:
            if handler_id not in self.handlers:
                return
            
            # Remove from topic handlers
            topics_to_remove = []
            for topic, handler_ids in self.topic_handlers.items():
                if handler_id in handler_ids:
                    handler_ids.remove(handler_id)
                    if not handler_ids:
                        topics_to_remove.append(topic)
            
            # Unsubscribe from empty topics
            for topic in topics_to_remove:
                await self.pubsub.unsubscribe(topic)
                del self.topic_handlers[topic]
            
            # Remove handler
            del self.handlers[handler_id]
            
            logger.info(f"Unsubscribed handler {handler_id}")
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe handler: {e}")
    
    async def send_direct_message(
        self, 
        message: AgentMessage, 
        recipient_id: str
    ) -> bool:
        """
        Send a direct message to a specific agent.
        
        Args:
            message: The message to send
            recipient_id: ID of the recipient agent
            
        Returns:
            True if message was sent successfully
        """
        try:
            # Set recipient
            message.recipient_id = recipient_id
            
            # Use agent-specific topic
            topic = f"agent.{recipient_id}"
            
            return await self.publish(message, topic)
            
        except Exception as e:
            logger.error(f"Failed to send direct message: {e}")
            return False
    
    async def broadcast_message(
        self, 
        message: AgentMessage, 
        exclude_agents: Optional[List[str]] = None
    ) -> bool:
        """
        Broadcast a message to all subscribed agents.
        
        Args:
            message: The message to broadcast
            exclude_agents: List of agent IDs to exclude from broadcast
            
        Returns:
            True if message was broadcast successfully
        """
        try:
            # Set broadcast flag
            message.broadcast = True
            
            # Use broadcast topic
            topic = "broadcast"
            
            # Include exclusion list in message context
            if exclude_agents:
                message.context["exclude_agents"] = exclude_agents
            
            return await self.publish(message, topic)
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return False
    
    async def request_response(
        self, 
        message: AgentMessage, 
        timeout: int = 30
    ) -> Optional[AgentMessage]:
        """
        Send a message and wait for a response.
        
        Args:
            message: The message to send
            timeout: Timeout in seconds
            
        Returns:
            The response message or None if timeout
        """
        try:
            # Set response required
            message.requires_response = True
            message.response_timeout = timeout
            
            # Create future for response
            response_future = asyncio.Future()
            self.pending_responses[message.message_id] = response_future
            
            # Send message
            success = await self.publish(message)
            if not success:
                del self.pending_responses[message.message_id]
                return None
            
            # Wait for response
            try:
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response
            except asyncio.TimeoutError:
                logger.warning(f"Request {message.message_id} timed out")
                return None
            finally:
                # Clean up
                if message.message_id in self.pending_responses:
                    del self.pending_responses[message.message_id]
            
        except Exception as e:
            logger.error(f"Failed to request response: {e}")
            return None
    
    async def _process_messages(self):
        """Process incoming messages from Redis."""
        try:
            while self.running:
                try:
                    # Get message from Redis
                    message = await self.pubsub.get_message(ignore_subscribe_messages=True)
                    
                    if not message:
                        await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                        continue
                    
                    # Parse message
                    if message['type'] == 'message':
                        await self._handle_message(message)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await asyncio.sleep(0.1)
        
        except asyncio.CancelledError:
            logger.info("Message processor cancelled")
        except Exception as e:
            logger.error(f"Message processor error: {e}")
    
    async def _handle_message(self, redis_message: Dict[str, Any]):
        """Handle a message received from Redis."""
        try:
            # Parse message data
            message_data = json.loads(redis_message['data'])
            message = AgentMessage(**message_data)
            
            # Get topic
            topic = redis_message['channel'].decode('utf-8')
            
            # Check if this is a response to a pending request
            if message.correlation_id and message.correlation_id in self.pending_responses:
                future = self.pending_responses[message.correlation_id]
                if not future.done():
                    future.set_result(message)
                return
            
            # Find handlers for this topic
            handler_ids = self.topic_handlers.get(topic, [])
            
            # Process message with each handler
            for handler_id in handler_ids:
                handler = self.handlers.get(handler_id)
                if not handler:
                    continue
                
                # Check if handler accepts this message type
                if message.message_type not in handler.message_types:
                    continue
                
                # Apply filter if present
                if handler.filter_func and not handler.filter_func(message):
                    continue
                
                # Check exclusion list for broadcasts
                if message.broadcast:
                    exclude_agents = message.context.get("exclude_agents", [])
                    if handler.agent_id in exclude_agents:
                        continue
                
                # Process message
                try:
                    await self._invoke_handler(handler, message)
                except Exception as e:
                    logger.error(f"Handler {handler_id} failed to process message: {e}")
            
        except Exception as e:
            logger.error(f"Failed to handle message: {e}")
    
    async def _invoke_handler(self, handler: MessageHandler, message: AgentMessage):
        """Invoke a message handler."""
        try:
            # Call handler callback
            if asyncio.iscoroutinefunction(handler.callback):
                await handler.callback(message)
            else:
                # Run in thread pool for sync handlers
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    handler.callback,
                    message
                )
            
        except Exception as e:
            logger.error(f"Handler callback failed: {e}")
            raise
    
    def _add_to_history(self, message: AgentMessage):
        """Add message to history with size management."""
        self.message_history.append(message)
        
        # Maintain history size
        if len(self.message_history) > self.max_history_size:
            self.message_history = self.message_history[-self.max_history_size:]
    
    def _track_response_required(self, message: AgentMessage):
        """Track messages that require responses."""
        # Set expiration time
        if message.response_timeout:
            expires_at = datetime.now() + timedelta(seconds=message.response_timeout)
            message.expires_at = expires_at
    
    async def get_message_history(
        self, 
        agent_id: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """
        Get message history with optional filtering.
        
        Args:
            agent_id: Filter by agent ID
            message_type: Filter by message type
            limit: Maximum number of messages to return
            
        Returns:
            List of messages matching the criteria
        """
        try:
            messages = self.message_history
            
            # Apply filters
            if agent_id:
                messages = [m for m in messages if m.sender_id == agent_id or m.recipient_id == agent_id]
            
            if message_type:
                messages = [m for m in messages if m.message_type == message_type]
            
            # Apply limit
            if limit > 0:
                messages = messages[-limit:]
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get message history: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get message broker statistics."""
        try:
            # Calculate message statistics
            total_messages = len(self.message_history)
            message_types = {}
            agent_activity = {}
            
            for message in self.message_history:
                # Count by message type
                msg_type = message.message_type.value
                message_types[msg_type] = message_types.get(msg_type, 0) + 1
                
                # Count by agent
                sender = message.sender_id
                agent_activity[sender] = agent_activity.get(sender, 0) + 1
            
            # Active handlers
            active_handlers = len(self.handlers)
            active_topics = len(self.topic_handlers)
            
            # Pending responses
            pending_responses = len(self.pending_responses)
            
            return {
                "total_messages": total_messages,
                "message_types": message_types,
                "agent_activity": agent_activity,
                "active_handlers": active_handlers,
                "active_topics": active_topics,
                "pending_responses": pending_responses,
                "broker_status": "running" if self.running else "stopped"
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    async def cleanup_expired_messages(self):
        """Clean up expired messages and pending responses."""
        try:
            now = datetime.now()
            
            # Clean up expired messages from history
            self.message_history = [
                msg for msg in self.message_history 
                if not msg.expires_at or msg.expires_at > now
            ]
            
            # Clean up expired pending responses
            expired_requests = []
            for msg_id, future in self.pending_responses.items():
                if future.done():
                    expired_requests.append(msg_id)
            
            for msg_id in expired_requests:
                del self.pending_responses[msg_id]
            
            logger.debug(f"Cleaned up {len(expired_requests)} expired requests")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired messages: {e}")


# Utility functions for creating common message handlers

def create_agent_handler(
    agent_id: str, 
    message_types: List[MessageType], 
    callback: Callable
) -> MessageHandler:
    """Create a message handler for an agent."""
    return MessageHandler(
        handler_id=f"agent_{agent_id}_{uuid.uuid4().hex[:8]}",
        agent_id=agent_id,
        message_types=set(message_types),
        callback=callback
    )


def create_filtered_handler(
    agent_id: str,
    message_types: List[MessageType],
    callback: Callable,
    filter_func: Callable[[AgentMessage], bool]
) -> MessageHandler:
    """Create a message handler with custom filtering."""
    return MessageHandler(
        handler_id=f"filtered_{agent_id}_{uuid.uuid4().hex[:8]}",
        agent_id=agent_id,
        message_types=set(message_types),
        callback=callback,
        filter_func=filter_func
    )


def create_priority_filter(min_priority: Priority) -> Callable[[AgentMessage], bool]:
    """Create a filter function for message priority."""
    def filter_func(message: AgentMessage) -> bool:
        return message.priority.value >= min_priority.value
    return filter_func


def create_sender_filter(allowed_senders: List[str]) -> Callable[[AgentMessage], bool]:
    """Create a filter function for allowed senders."""
    def filter_func(message: AgentMessage) -> bool:
        return message.sender_id in allowed_senders
    return filter_func
