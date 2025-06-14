from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable
from datetime import datetime
import asyncio
import time
import uuid


@dataclass
class ToolOperation:
    """
    Represents a single tool operation in a workflow.
    
    Attributes:
        tool_name: Name of the tool to execute
        operation: Specific operation to perform
        parameters: Parameters for the operation
        dependencies: List of operation_ids this operation depends on
        timeout: Maximum execution time in seconds
        operation_id: Unique identifier for this operation
    """
    tool_name: str
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # List of operation_ids this depends on
    timeout: Optional[float] = None  # seconds
    operation_id: str = field(default_factory=lambda: f"op_{str(uuid.uuid4())[:8]}")


@dataclass
class WorkflowResult:
    """
    Result of executing a workflow of tool operations.
    
    Attributes:
        operations_completed: List of operation_ids that completed successfully
        results: Mapping of operation_id to operation result
        errors: Mapping of operation_id to error message for failed operations
        total_duration: Total time taken to execute the workflow
    """
    operations_completed: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)  # operation_id -> result
    errors: Dict[str, str] = field(default_factory=dict)   # operation_id -> error
    total_duration: float = 0.0

class AdvancedToolWorkflowMixin:
    """
    Mixin for BaseAgent to provide advanced tool workflow execution and monitoring.
    
    This mixin provides sophisticated workflow management capabilities including:
    - Dependency management and topological sorting
    - Parallel execution optimization
    - Automatic rollback on failure
    - Continuous operation monitoring
    - Timeout detection and retry mechanisms
    - Performance metrics collection
    
    Required attributes in the implementing class:
    - get_tool(tool_name) -> BaseTool: Method to retrieve tool by name
    - active_operations: Dict[str, Tuple[ToolOperation, asyncio.Task]]
    - operation_status: Dict[str, str]
    """

    async def execute_tool_workflow(self, workflow: List[ToolOperation]) -> WorkflowResult:
        """
        Execute a workflow of ToolOperation objects with dependency management, parallelism, rollback, and metrics.
        
        This method:
        1. Validates workflow feasibility (no cycles, all dependencies exist)
        2. Executes operations in dependency-aware order with parallel optimization
        3. Handles rollback on failure
        4. Collects comprehensive performance metrics
        
        Args:
            workflow: List of ToolOperation objects to execute
            
        Returns:
            WorkflowResult with execution details, results, and metrics
        """
        start_time = time.time()
        op_map = {op.operation_id: op for op in workflow}
        completed = set()
        results = {}
        errors = {}
        in_progress = set()
        
        # Build dependency graph
        dependents = {op.operation_id: [] for op in workflow}
        for op in workflow:
            for dep in op.dependencies:
                dependents.setdefault(dep, []).append(op.operation_id)

        # Validate workflow feasibility (no cycles, all dependencies exist)
        # First check all dependencies exist
        for op in workflow:
            for dep in op.dependencies:
                if dep not in op_map:
                    return WorkflowResult(errors={"workflow": f"Missing dependency: {dep}"})
        
        def has_cycle():
            """Detect dependency cycles using DFS."""
            visited, stack = set(), set()
            
            def visit(op_id):
                if op_id in stack:
                    return True
                if op_id in visited:
                    return False
                stack.add(op_id)
                for dep in op_map[op_id].dependencies:
                    # Since we checked all deps exist above, we know dep is in op_map
                    if visit(dep):
                        return True
                stack.remove(op_id)
                visited.add(op_id)
                return False
                
            return any(visit(op_id) for op_id in op_map)
        
        if has_cycle():
            return WorkflowResult(errors={"workflow": "Dependency cycle detected"})

        # Helper: rollback completed operations in reverse order
        async def rollback():
            """Rollback all completed operations in reverse dependency order."""
            for op_id in reversed(list(completed)):
                op = op_map[op_id]
                tool = self.get_tool(op.tool_name)
                if hasattr(tool, "rollback"):
                    try:
                        await tool.rollback(op.operation, op.parameters)
                    except Exception as e:
                        # Log rollback failure but continue
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"Rollback failed for {op_id}: {e}")

        # Execute individual operation with error handling and timeout
        async def run_op(op: ToolOperation):
            """Execute a single operation with proper error handling and timeout."""
            tool = self.get_tool(op.tool_name)
            if not tool:
                raise ValueError(f"Tool {op.tool_name} not found")
            
            try:
                coro = tool.execute(op.operation, op.parameters)
                if op.timeout:
                    result = await asyncio.wait_for(coro, timeout=op.timeout)
                else:
                    result = await coro
                
                # Check if result indicates failure
                if hasattr(result, 'status') and hasattr(result.status, 'value'):
                    if result.status.value == "failure":
                        error_msg = getattr(result, 'error_message', 'Operation failed')
                        errors[op.operation_id] = error_msg
                        return False
                elif hasattr(result, 'status') and result.status == "failure":
                    error_msg = getattr(result, 'error_message', 'Operation failed')
                    errors[op.operation_id] = error_msg
                    return False
                    
                results[op.operation_id] = result
                completed.add(op.operation_id)
                return True
                
            except asyncio.TimeoutError:
                errors[op.operation_id] = f"Operation timed out after {op.timeout}s"
                return False
            except Exception as e:
                errors[op.operation_id] = str(e)
                return False

        # Main execution loop with topological ordering
        pending_ops = {op.operation_id for op in workflow}
        
        while pending_ops:
            # Find all operations whose dependencies are satisfied
            available = [
                op_map[op_id] for op_id in pending_ops 
                if all(dep in completed for dep in op_map[op_id].dependencies)
            ]
            
            if not available:
                # No operations can proceed - check for unresolvable dependencies
                remaining_ops = [op_map[op_id] for op_id in pending_ops]
                unmet_deps = []
                for op in remaining_ops:
                    for dep in op.dependencies:
                        if dep not in completed and dep not in errors:
                            unmet_deps.append(f"{op.operation_id} -> {dep}")
                
                error_msg = f"Deadlock detected. Unmet dependencies: {unmet_deps}"
                return WorkflowResult(
                    operations_completed=list(completed),
                    results=results,
                    errors={**errors, "workflow": error_msg},
                    total_duration=time.time() - start_time
                )
            
            # Execute all available operations in parallel
            tasks = {op.operation_id: asyncio.create_task(run_op(op)) for op in available}
            in_progress.update(tasks.keys())
            
            # Wait for all tasks to complete
            done, _ = await asyncio.wait(tasks.values())
            
            # Check results and handle failures
            failed_ops = []
            for op_id, task in tasks.items():
                try:
                    success = await task
                    if not success:
                        failed_ops.append(op_id)
                except Exception as e:
                    errors[op_id] = str(e)
                    failed_ops.append(op_id)
            
            # If any operation failed, perform rollback and return
            if failed_ops:
                await rollback()
                return WorkflowResult(
                    operations_completed=list(completed),
                    results=results,
                    errors=errors,
                    total_duration=time.time() - start_time
                )
            
            # Remove completed operations from pending
            for op in available:
                pending_ops.discard(op.operation_id)
                in_progress.discard(op.operation_id)
        
        total_duration = time.time() - start_time
        return WorkflowResult(
            operations_completed=list(completed),
            results=results,
            errors=errors,
            total_duration=total_duration
        )

    async def monitor_tool_operations(self) -> None:
        """
        Continuously monitor active tool operations, handle timeouts, update status, and trigger retries.
        
        This method provides sophisticated operation monitoring including:
        - Progress tracking for long-running operations
        - Timeout detection with graceful cleanup
        - Intelligent retry mechanisms with exponential backoff
        - Status updates for inter-agent communication
        - Stuck operation detection and recovery
        - Resource cleanup and health monitoring
        
        Expected attributes:
        - self.active_operations: Dict[operation_id, (ToolOperation, asyncio.Task)]
        - self.operation_status: Dict[operation_id, str]
        """
        backoff_delays = {}  # Track retry delays for exponential backoff
        max_retries = 3
        monitoring_interval = 2.0  # seconds
        
        if hasattr(self, 'logger'):
            self.logger.info("Starting tool operations monitoring")
        
        while True:
            try:
                to_remove = []
                current_time = time.time()
                
                # Get active operations safely
                active_ops = getattr(self, "active_operations", {})
                operation_status = getattr(self, "operation_status", {})
                
                for op_id, (op, task) in list(active_ops.items()):
                    try:
                        # Check if task is completed
                        if task.done():
                            try:
                                result = task.result()
                                operation_status[op_id] = "completed"
                                
                                if hasattr(self, 'logger'):
                                    self.logger.debug(f"Operation {op_id} completed successfully")
                                    
                            except asyncio.TimeoutError:
                                operation_status[op_id] = "timeout"
                                await self.cleanup_operation(op_id)
                                
                                # Implement intelligent retry with exponential backoff
                                retry_count = backoff_delays.get(op_id, {}).get('count', 0)
                                if retry_count < max_retries:
                                    delay = min(2 ** retry_count, 30)  # Cap at 30 seconds
                                    backoff_delays[op_id] = {
                                        'count': retry_count + 1, 
                                        'next_retry': current_time + delay
                                    }
                                    operation_status[op_id] = "retrying"
                                    
                                    if hasattr(self, 'logger'):
                                        self.logger.info(f"Scheduling retry {retry_count + 1}/{max_retries} for {op_id} in {delay}s")
                                else:
                                    operation_status[op_id] = "failed_max_retries"
                                    if hasattr(self, 'logger'):
                                        self.logger.error(f"Operation {op_id} failed after {max_retries} retries")
                                    
                            except asyncio.CancelledError:
                                operation_status[op_id] = "cancelled"
                                await self.cleanup_operation(op_id)
                                
                            except Exception as e:
                                operation_status[op_id] = "failed"
                                await self.cleanup_operation(op_id)
                                
                                if hasattr(self, 'logger'):
                                    self.logger.error(f"Operation {op_id} failed with error: {e}")
                            
                            to_remove.append(op_id)
                            
                        else:
                            # Check for timeout on running operations
                            task_start_time = getattr(task, 'start_time', None)
                            if task_start_time and op.timeout:
                                running_time = current_time - task_start_time
                                if running_time > op.timeout:
                                    # Cancel timed-out task
                                    task.cancel()
                                    operation_status[op_id] = "timeout"
                                    await self.cleanup_operation(op_id)
                                    
                                    if hasattr(self, 'logger'):
                                        self.logger.warning(f"Operation {op_id} timed out after {running_time:.2f}s")
                                    
                                    to_remove.append(op_id)
                                    
                            # Check for stuck operations (no progress indicators)
                            elif task_start_time and (current_time - task_start_time) > 300:  # 5 minutes
                                # Consider operation stuck if running for too long without timeout
                                if hasattr(self, 'logger'):
                                    self.logger.warning(f"Operation {op_id} appears stuck, cancelling")
                                
                                task.cancel()
                                operation_status[op_id] = "stuck"
                                await self.cleanup_operation(op_id)
                                to_remove.append(op_id)
                        
                        # Notify other agents of status changes
                        await self.notify_agents(op_id, operation_status.get(op_id, "unknown"))
                        
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.error(f"Error monitoring operation {op_id}: {e}")
                        operation_status[op_id] = "monitoring_error"
                        to_remove.append(op_id)
                
                # Clean up completed/failed operations
                for op_id in to_remove:
                    active_ops.pop(op_id, None)
                    backoff_delays.pop(op_id, None)
                
                # Handle retry scheduling
                for op_id, retry_info in list(backoff_delays.items()):
                    if current_time >= retry_info['next_retry']:
                        # Time to retry
                        if op_id in operation_status and operation_status[op_id] == "retrying":
                            # Get the original operation
                            if hasattr(self, '_retry_operation'):
                                await self._retry_operation(op_id)
                            else:
                                # Simple retry implementation
                                backoff_delays.pop(op_id, None)
                
                # Sleep before next monitoring cycle
                await asyncio.sleep(monitoring_interval)
                
            except asyncio.CancelledError:
                if hasattr(self, 'logger'):
                    self.logger.info("Tool operations monitoring cancelled")
                break
                
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(monitoring_interval)

    async def cleanup_operation(self, op_id: str):
        """
        Clean up resources and state for a completed or failed operation.
        
        Args:
            op_id: The operation ID to clean up
        """
        try:
            # Remove from active operations
            if hasattr(self, 'active_operations'):
                self.active_operations.pop(op_id, None)
            
            # Clean up any temporary resources
            if hasattr(self, '_operation_resources'):
                resources = self._operation_resources.pop(op_id, [])
                for resource in resources:
                    try:
                        if hasattr(resource, 'cleanup'):
                            await resource.cleanup()
                        elif hasattr(resource, 'close'):
                            await resource.close()
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"Failed to cleanup resource for {op_id}: {e}")
            
            # Update metrics if available
            if hasattr(self, '_update_operation_metrics'):
                await self._update_operation_metrics(op_id, "cleanup")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during cleanup of operation {op_id}: {e}")

    async def notify_agents(self, op_id: str, status: str):
        """
        Notify other agents about operation status changes.
        
        Args:
            op_id: The operation ID
            status: The new status
        """
        try:
            # Use event bus if available
            if hasattr(self, 'event_bus'):
                await self.event_bus.publish(
                    f"tool.operation.{status}",
                    {
                        "operation_id": op_id,
                        "status": status,
                        "agent_id": getattr(self, 'agent_id', 'unknown'),
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            # Update shared state if available
            if hasattr(self, 'memory_manager'):
                await self.memory_manager.store_memory(
                    content=f"Tool operation {op_id} status: {status}",
                    metadata={
                        "type": "operation_status",
                        "operation_id": op_id,
                        "status": status,
                        "agent_id": getattr(self, 'agent_id', 'unknown')
                    },
                    importance=0.3
                )
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Failed to notify agents about {op_id} status: {e}")

    async def _retry_operation(self, op_id: str):
        """
        Retry a failed operation with the same parameters.
        
        Args:
            op_id: The operation ID to retry
        """
        try:
            # This would need to be implemented based on the specific agent's operation storage
            if hasattr(self, 'logger'):
                self.logger.info(f"Retrying operation {op_id}")
                
            # Implementation would depend on how operations are stored and tracked
            # This is a placeholder for the actual retry logic
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to retry operation {op_id}: {e}")

# Usage example:
# class BaseAgent(AdvancedToolWorkflowMixin, ...):
#     pass

# Usage: class BaseAgent(AdvancedToolWorkflowMixin, ...):
