"""Pensieve FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import init_db
from app.routers import (
    analysis_router,
    auth_router,
    concepts_router,
    entries_router,
    reflections_router,
)

# Configure logging without logging raw journal contents
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pensieve")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    logger.info("Starting Pensieve Backend...")
    # Initialize SQLite database tables
    init_db()
    logger.info("Database tables initialized.")
    yield
    logger.info("Shutting down Pensieve Backend.")


app = FastAPI(
    title=settings.APP_NAME,
    description="Privacy-first AI journaling platform with grounded psychological & philosophical reflection.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for React frontend (Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers under /api
app.include_router(auth_router, prefix="/api")
app.include_router(entries_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
app.include_router(reflections_router, prefix="/api")
app.include_router(concepts_router, prefix="/api")


@app.get("/", tags=["Health"])
def root():
    """Root status endpoint."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "message": "Welcome to the Pensieve API.",
    }


@app.get("/api/health", tags=["Health"])
def health_check():
    """Health check endpoint for frontend and monitoring."""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "database": "connected",
        "ml_pipeline": "ready",
    }
