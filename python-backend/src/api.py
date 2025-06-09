"""
FastAPI Application for FlutterSwarm Multi-Agent System.

This module provides the REST API interface for interacting with the
FlutterSwarm multi-agent development system.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .system import flutter_swarm_system, initialize_system, start_system, stop_system
from .config.settings import settings
from .config import get_logger

logger = get_logger("api")


# Request/Response Models
class TaskRequest(BaseModel):
    """Request model for task submission."""
    description: str = Field(..., description="Task description")
    task_type: str = Field(default="analysis", description="Type of task")
    priority: str = Field(default="normal", description="Task priority")
    requirements: List[Dict[str, Any]] = Field(default_factory=list)
    expected_deliverables: List[Dict[str, Any]] = Field(default_factory=list)
    project_context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    """Response model for task results."""
    success: bool
    task_id: str
    status: str
    result: Dict[str, Any]
    deliverables: Dict[str, Any]
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any]


class SystemStatus(BaseModel):
    """System status response model."""
    system: Dict[str, Any]
    agents: Dict[str, Any]
    event_bus: Dict[str, Any]
    llm_client: Dict[str, Any]


class AgentRegistration(BaseModel):
    """Agent registration request model."""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    specializations: List[str] = Field(default_factory=list)
    max_concurrent_tasks: int = Field(default=5)


# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting FlutterSwarm API...")
    
    try:
        # Initialize and start the system
        await initialize_system()
        await start_system()
        logger.info("FlutterSwarm system started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start FlutterSwarm system: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down FlutterSwarm system...")
        try:
            await stop_system()
            logger.info("FlutterSwarm system stopped successfully")
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="FlutterSwarm Multi-Agent System",
    description="Production-grade multi-agent system for Flutter app development",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get system instance
async def get_system():
    """Dependency to get the current system instance."""
    if not flutter_swarm_system.is_running:
        raise HTTPException(status_code=503, detail="FlutterSwarm system is not running")
    return flutter_swarm_system


# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "FlutterSwarm Multi-Agent System API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check(system=Depends(get_system)):
    """System health check endpoint."""
    try:
        status = await system.get_system_status()
        return {
            "status": "healthy" if system.is_running else "unhealthy",
            "details": status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/status", response_model=SystemStatus)
async def get_system_status(system=Depends(get_system)):
    """Get detailed system status."""
    try:
        status = await system.get_system_status()
        return SystemStatus(**status)
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@app.post("/tasks", response_model=TaskResponse)
async def submit_task(
    task_request: TaskRequest,
    background_tasks: BackgroundTasks,
    system=Depends(get_system)
):
    """Submit a task for processing."""
    try:
        logger.info(f"Received task request: {task_request.description[:100]}...")
        
        # Convert request to internal format
        request_dict = {
            "description": task_request.description,
            "task_type": task_request.task_type,
            "priority": task_request.priority,
            "requirements": task_request.requirements,
            "expected_deliverables": task_request.expected_deliverables,
            "project_context": task_request.project_context,
            "metadata": task_request.metadata
        }
        
        # Process request through the system
        result = await system.process_request(request_dict)
        
        return TaskResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to process task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process task: {str(e)}")


@app.get("/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task_status(task_id: str, system=Depends(get_system)):
    """Get status of a specific task."""
    try:
        # This would typically query a database or task tracker
        # For now, return a placeholder response
        return {
            "task_id": task_id,
            "status": "completed",
            "message": "Task status tracking not yet implemented"
        }
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@app.get("/agents", response_model=Dict[str, Any])
async def list_agents(system=Depends(get_system)):
    """List all registered agents."""
    try:
        agents_info = {}
        for agent_id, agent in system.agents.items():
            agents_info[agent_id] = await agent.get_status()
        
        return {
            "agents": agents_info,
            "total_count": len(agents_info)
        }
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")


@app.get("/agents/{agent_id}", response_model=Dict[str, Any])
async def get_agent_status(agent_id: str, system=Depends(get_system)):
    """Get status of a specific agent."""
    try:
        if agent_id not in system.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        agent = system.agents[agent_id]
        return await agent.get_status()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")


@app.post("/agents/register", response_model=Dict[str, str])
async def register_agent(
    registration: AgentRegistration,
    system=Depends(get_system)
):
    """Register a new agent with the system."""
    try:
        # This would typically create and register a new agent instance
        # For now, return a placeholder response
        return {
            "message": f"Agent registration received for {registration.agent_id}",
            "status": "Agent registration not yet implemented"
        }
    except Exception as e:
        logger.error(f"Failed to register agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register agent: {str(e)}")


@app.delete("/agents/{agent_id}", response_model=Dict[str, str])
async def unregister_agent(agent_id: str, system=Depends(get_system)):
    """Unregister an agent from the system."""
    try:
        await system.unregister_agent(agent_id)
        return {"message": f"Agent {agent_id} unregistered successfully"}
    except Exception as e:
        logger.error(f"Failed to unregister agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unregister agent: {str(e)}")


@app.get("/memory/{agent_id}", response_model=Dict[str, Any])
async def get_agent_memory_stats(agent_id: str, system=Depends(get_system)):
    """Get memory statistics for a specific agent."""
    try:
        if agent_id not in system.memory_managers:
            raise HTTPException(status_code=404, detail=f"Memory manager for agent {agent_id} not found")
        
        memory_manager = system.memory_managers[agent_id]
        return memory_manager.get_statistics()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")


@app.get("/events/metrics", response_model=Dict[str, Any])
async def get_event_metrics(system=Depends(get_system)):
    """Get event bus metrics."""
    try:
        if not system.event_bus:
            raise HTTPException(status_code=404, detail="Event bus not available")
        
        return await system.event_bus.get_metrics()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get event metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get event metrics: {str(e)}")


@app.get("/llm/stats", response_model=Dict[str, Any])
async def get_llm_stats(system=Depends(get_system)):
    """Get LLM client statistics."""
    try:
        if not system.llm_client:
            raise HTTPException(status_code=404, detail="LLM client not available")
        
        return system.llm_client.get_statistics()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get LLM stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get LLM stats: {str(e)}")


@app.post("/system/shutdown", response_model=Dict[str, str])
async def shutdown_system(background_tasks: BackgroundTasks, system=Depends(get_system)):
    """Shutdown the FlutterSwarm system gracefully."""
    try:
        # Schedule shutdown in background to allow response to be sent
        background_tasks.add_task(stop_system)
        
        return {"message": "System shutdown initiated"}
        
    except Exception as e:
        logger.error(f"Failed to shutdown system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to shutdown system: {str(e)}")


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


# Development server
def run_development_server():
    """Run the development server."""
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    run_development_server()
