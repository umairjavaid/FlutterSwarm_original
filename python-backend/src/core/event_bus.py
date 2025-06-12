"""
Event Bus System for FlutterSwarm Multi-Agent Communication.

This module implements a publish-subscribe event bus that enables
decoupled communication between agents in the system.
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from ..models.agent_models import AgentMessage, CommunicationError


logger = logging.getLogger(__name__)


@dataclass
class EventSubscription:
    """
    Represents a subscription to an event topic.
    
    Tracks the subscriber information and callback function
    for handling events.
    """
    subscriber_id: str
    topic: str
    callback: Callable[[AgentMessage], Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    message_count: int = 0
    last_message_at: Optional[datetime] = None
    
    async def handle_message(self, message: AgentMessage) -> None:
        """Handle an incoming message for this subscription."""
        try:
            self.message_count += 1
            self.last_message_at = datetime.utcnow()
            
            # Call the callback function
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(message)
            else:
                self.callback(message)
                
        except Exception as e:
            logger.error(f"Error handling message in subscription {self.subscriber_id}: {e}")
            raise


class EventBus:
    """
    Central event bus for agent communication.
    
    This class implements a publish-subscribe pattern that allows agents
    to communicate without direct coupling. Messages are routed based on
    topics and patterns.
    
    Features:
    - Topic-based subscriptions
    - Pattern matching for topic subscriptions
    - Message queuing and delivery guarantees
    - Dead letter queue for failed message deliveries
    - Event auditing and logging
    - Health monitoring
    
    Attributes:
        subscriptions: Dictionary mapping topics to subscriber lists
        message_queue: Queue for pending message deliveries
        dead_letter_queue: Queue for failed message deliveries
        event_history: History of recent events for debugging
        metrics: Health and delivery metrics
    """
    
    def __init__(self, max_history: int = None, max_retry_attempts: int = None):
        """
        Initialize the event bus.
        
        Args:
            max_history: Maximum number of events to keep in history
            max_retry_attempts: Maximum retry attempts for failed deliveries
        """
        from ..config import settings
        
        # Use config values if not explicitly provided
        if max_history is None:
            max_history = settings.event_bus.max_history
        if max_retry_attempts is None:
            max_retry_attempts = settings.event_bus.max_retry_attempts
        self.subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self.pattern_subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.dead_letter_queue: asyncio.Queue = asyncio.Queue()
        self.event_history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.max_retry_attempts = max_retry_attempts
        
        # Metrics tracking
        self.metrics = {
            "total_messages": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "active_subscriptions": 0,
            "average_delivery_time": 0.0
        }
        
        # Background tasks
        self._message_processor_task: Optional[asyncio.Task] = None
        self._start_message_processor()
        
        logger.info("Event bus initialized")
    
    def _start_message_processor(self) -> None:
        """Start the background message processor task."""
        self._message_processor_task = asyncio.create_task(self._process_messages())
    
    async def _process_messages(self) -> None:
        """
        Background task to process queued messages.
        
        This ensures that message delivery doesn't block the publisher
        and provides retry capabilities for failed deliveries.
        """
        while True:
            try:
                # Get message from queue with timeout
                try:
                    message_data = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                start_time = datetime.utcnow()
                success = await self._deliver_message(message_data)
                
                # Update metrics
                delivery_time = (datetime.utcnow() - start_time).total_seconds()
                self._update_delivery_metrics(success, delivery_time)
                
                # Mark task as done
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                await asyncio.sleep(1)
    
    async def _deliver_message(self, message_data: Dict[str, Any]) -> bool:
        """
        Deliver a message to its subscribers.
        
        Args:
            message_data: Dictionary containing message and delivery info
        
        Returns:
            True if delivery was successful, False otherwise
        """
        message = message_data["message"]
        topic = message_data["topic"]
        retry_count = message_data.get("retry_count", 0)
        
        try:
            # Get subscribers for exact topic match
            subscribers = self.subscriptions.get(topic, [])
            
            # Get subscribers for pattern matches
            pattern_subscribers = []
            for pattern, subs in self.pattern_subscriptions.items():
                if self._topic_matches_pattern(topic, pattern):
                    pattern_subscribers.extend(subs)
            
            all_subscribers = subscribers + pattern_subscribers
            
            if not all_subscribers:
                logger.debug(f"No subscribers for topic: {topic}")
                return True
            
            # Deliver to all subscribers
            delivery_tasks = []
            for subscription in all_subscribers:
                task = asyncio.create_task(subscription.handle_message(message))
                delivery_tasks.append(task)
            
            # Wait for all deliveries to complete
            results = await asyncio.gather(*delivery_tasks, return_exceptions=True)
            
            # Check for failures
            failed_deliveries = [r for r in results if isinstance(r, Exception)]
            
            if failed_deliveries:
                logger.error(f"Failed deliveries for topic {topic}: {len(failed_deliveries)}")
                
                # Retry if not exceeded max attempts
                if retry_count < self.max_retry_attempts:
                    message_data["retry_count"] = retry_count + 1
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    await self.message_queue.put(message_data)
                    return False
                else:
                    # Move to dead letter queue
                    await self.dead_letter_queue.put({
                        "message": message,
                        "topic": topic,
                        "failures": failed_deliveries,
                        "retry_count": retry_count,
                        "timestamp": datetime.utcnow()
                    })
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error delivering message to topic {topic}: {e}")
            return False
    
    def _topic_matches_pattern(self, topic: str, pattern: str) -> bool:
        """
        Check if a topic matches a subscription pattern.
        
        Supports wildcards:
        - * matches any single word
        - ** matches any number of words
        - # matches everything after this point
        
        Args:
            topic: The topic to check
            pattern: The pattern to match against
        
        Returns:
            True if topic matches pattern
        """
        if pattern == "#":
            return True
        
        topic_parts = topic.split(".")
        pattern_parts = pattern.split(".")
        
        # Simple implementation - can be enhanced for more complex patterns
        if "**" in pattern:
            # Handle ** wildcard
            return True  # Simplified for now
        
        if len(topic_parts) != len(pattern_parts):
            return False
        
        for topic_part, pattern_part in zip(topic_parts, pattern_parts):
            if pattern_part != "*" and pattern_part != topic_part:
                return False
        
        return True
    
    def _update_delivery_metrics(self, success: bool, delivery_time: float) -> None:
        """Update delivery metrics."""
        self.metrics["total_messages"] += 1
        
        if success:
            self.metrics["successful_deliveries"] += 1
        else:
            self.metrics["failed_deliveries"] += 1
        
        # Update average delivery time
        total_successful = self.metrics["successful_deliveries"]
        if total_successful > 1:
            current_avg = self.metrics["average_delivery_time"]
            self.metrics["average_delivery_time"] = (
                (current_avg * (total_successful - 1) + delivery_time) / total_successful
            )
        else:
            self.metrics["average_delivery_time"] = delivery_time
    
    async def subscribe(
        self,
        topic: str,
        callback: Callable[[AgentMessage], Any],
        subscriber_id: Optional[str] = None
    ) -> str:
        """
        Subscribe to events on a specific topic.
        
        Args:
            topic: Topic to subscribe to (can include wildcards)
            callback: Function to call when events are received
            subscriber_id: Optional subscriber ID (generated if not provided)
        
        Returns:
            Subscription ID for managing the subscription
        
        Raises:
            CommunicationError: If subscription fails
        """
        try:
            if subscriber_id is None:
                subscriber_id = str(uuid4())
            
            subscription = EventSubscription(
                subscriber_id=subscriber_id,
                topic=topic,
                callback=callback
            )
            
            # Check if topic contains wildcards
            if "*" in topic or "#" in topic:
                self.pattern_subscriptions[topic].append(subscription)
            else:
                self.subscriptions[topic].append(subscription)
            
            self.metrics["active_subscriptions"] += 1
            
            logger.info(f"Subscriber {subscriber_id} subscribed to topic: {topic}")
            return subscriber_id
            
        except Exception as e:
            logger.error(f"Failed to subscribe to topic {topic}: {e}")
            raise CommunicationError(f"Subscription failed: {e}")
    
    async def unsubscribe(self, topic: str, subscriber_id: str) -> bool:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: Topic to unsubscribe from
            subscriber_id: ID of the subscriber
        
        Returns:
            True if unsubscription was successful
        """
        try:
            # Check exact topic subscriptions
            if topic in self.subscriptions:
                self.subscriptions[topic] = [
                    sub for sub in self.subscriptions[topic]
                    if sub.subscriber_id != subscriber_id
                ]
                
                # Remove empty topic entries
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
            
            # Check pattern subscriptions
            if topic in self.pattern_subscriptions:
                self.pattern_subscriptions[topic] = [
                    sub for sub in self.pattern_subscriptions[topic]
                    if sub.subscriber_id != subscriber_id
                ]
                
                if not self.pattern_subscriptions[topic]:
                    del self.pattern_subscriptions[topic]
            
            self.metrics["active_subscriptions"] -= 1
            logger.info(f"Subscriber {subscriber_id} unsubscribed from topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from topic {topic}: {e}")
            return False
    
    async def publish(self, topic: str, message: AgentMessage) -> None:
        """
        Publish a message to a topic.
        
        Args:
            topic: Topic to publish to
            message: Message to publish
        
        Raises:
            CommunicationError: If publishing fails
        """
        try:
            # Add to event history
            event_record = {
                "topic": topic,
                "message": message.to_dict(),
                "timestamp": datetime.utcnow(),
                "event_id": str(uuid4())
            }
            
            self.event_history.append(event_record)
            
            # Trim history if needed
            if len(self.event_history) > self.max_history:
                self.event_history = self.event_history[-self.max_history:]
            
            # Queue message for delivery
            message_data = {
                "message": message,
                "topic": topic,
                "retry_count": 0,
                "published_at": datetime.utcnow()
            }
            
            await self.message_queue.put(message_data)
            
            logger.debug(f"Message published to topic: {topic}")
            
        except Exception as e:
            logger.error(f"Failed to publish message to topic {topic}: {e}")
            raise CommunicationError(f"Publishing failed: {e}")
    
    async def get_subscriber_count(self, topic: str) -> int:
        """Get the number of subscribers for a topic."""
        count = len(self.subscriptions.get(topic, []))
        
        # Add pattern subscribers
        for pattern in self.pattern_subscriptions:
            if self._topic_matches_pattern(topic, pattern):
                count += len(self.pattern_subscriptions[pattern])
        
        return count
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics and health information."""
        return {
            **self.metrics,
            "queue_size": self.message_queue.qsize(),
            "dead_letter_queue_size": self.dead_letter_queue.qsize(),
            "active_topics": len(self.subscriptions) + len(self.pattern_subscriptions),
            "event_history_size": len(self.event_history)
        }
    
    async def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent event history."""
        return self.event_history[-limit:]
    
    async def health_check(self) -> bool:
        """Perform health check on the event bus."""
        try:
            # Check if message processor is running
            if self._message_processor_task is None or self._message_processor_task.done():
                logger.error("Message processor task is not running")
                return False
            
            # Check queue sizes
            if self.message_queue.qsize() > self.max_history:
                logger.warning("Message queue is getting large")
            
            if self.dead_letter_queue.qsize() > (self.max_history // 10):  # 10% of max history
                logger.warning("Dead letter queue has many failed messages")
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the event bus and cleanup resources."""
        logger.info("Shutting down event bus")
        
        try:
            # Cancel message processor
            if self._message_processor_task:
                self._message_processor_task.cancel()
                try:
                    await self._message_processor_task
                except asyncio.CancelledError:
                    pass
            
            # Wait for pending messages to be processed
            await self.message_queue.join()
            
            # Clear subscriptions
            self.subscriptions.clear()
            self.pattern_subscriptions.clear()
            
            logger.info("Event bus shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during event bus shutdown: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self._message_processor_task and not self._message_processor_task.done():
            self._message_processor_task.cancel()
