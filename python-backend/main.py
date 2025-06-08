"""
FlutterSwarm - Multi-Agent Flutter Development System
Main entry point for the Python backend service.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import Settings
from core.logging_config import setup_logging
from api.main import create_app
from services.backend_service import BackendService

logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FlutterSwarm Backend Service")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the server to (default: localhost)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to configuration file"
    )
    
    return parser.parse_args()

def setup_app(settings: Settings) -> FastAPI:
    """Set up the FastAPI application."""
    app = create_app(settings)
    
    # Add CORS middleware for VS Code extension
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, be more specific
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app

async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        # Load settings
        settings = Settings(
            config_file=args.config_file,
            port=args.port,
            host=args.host,
            log_level=args.log_level
        )
        
        logger.info(f"Starting FlutterSwarm Backend Service...")
        logger.info(f"Host: {settings.host}")
        logger.info(f"Port: {settings.port}")
        logger.info(f"Log Level: {settings.log_level}")
        
        # Create FastAPI app
        app = setup_app(settings)
        
        # Initialize backend service
        backend_service = BackendService(settings)
        app.state.backend_service = backend_service
        
        # Start the backend service
        await backend_service.start()
        
        # Configure uvicorn
        config = uvicorn.Config(
            app=app,
            host=settings.host,
            port=settings.port,
            reload=args.reload,
            log_level=args.log_level.lower(),
            access_log=True
        )
        
        # Create and run server
        server = uvicorn.Server(config)
        
        # Handle graceful shutdown
        try:
            await server.serve()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            logger.info("Shutting down backend service...")
            await backend_service.stop()
            logger.info("Backend service stopped")
            
    except Exception as e:
        logger.error(f"Failed to start backend service: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Handle both sync and async execution
    if sys.platform == "win32":
        # Windows-specific event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())
