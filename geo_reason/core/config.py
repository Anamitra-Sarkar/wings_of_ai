"""
Configuration management for Geo-Reason system.

Uses Pydantic for validation and environment variable loading.
Supports enterprise-level configuration with secrets management.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Enterprise configuration settings for Geo-Reason.
    
    Environment variables can be set directly or via .env file.
    All secrets should be provided via environment variables in production.
    """
    
    # Application Settings
    app_name: str = "Geo-Reason"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Model Settings
    model_name: str = Field(
        default="meta-llama/Llama-3.2-3B-Instruct",
        description="HuggingFace model name for the LLM"
    )
    model_device: str = Field(
        default="auto",
        description="Device for model inference (cuda, cpu, auto)"
    )
    model_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for generation"
    )
    model_temperature: float = Field(
        default=0.1,
        description="Temperature for generation (lower = more deterministic)"
    )
    use_quantization: bool = Field(
        default=True,
        description="Use 4-bit quantization for memory efficiency"
    )
    
    # RAG Settings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model for document embeddings"
    )
    vector_store_path: Path = Field(
        default=Path("./data/vector_store"),
        description="Path to ChromaDB vector store"
    )
    chunk_size: int = Field(
        default=1000,
        description="Document chunk size for embedding"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between document chunks"
    )
    retrieval_top_k: int = Field(
        default=5,
        description="Number of documents to retrieve"
    )
    
    # Knowledge Base Settings
    docs_path: Path = Field(
        default=Path("./data/docs"),
        description="Path to documentation files"
    )
    supported_doc_types: List[str] = Field(
        default=["pdf", "md", "txt", "docx"],
        description="Supported document types"
    )
    
    # Sandbox Settings
    sandbox_timeout: int = Field(
        default=300,
        description="Maximum execution time in seconds"
    )
    sandbox_memory_limit: str = Field(
        default="4G",
        description="Maximum memory for sandbox execution"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retries for failed operations"
    )
    
    # Data Source Settings
    osm_cache_dir: Path = Field(
        default=Path("./data/osm_cache"),
        description="Cache directory for OSM data"
    )
    bhoonidhi_api_url: Optional[str] = Field(
        default=None,
        description="Bhoonidhi API endpoint URL"
    )
    bhoonidhi_api_key: Optional[str] = Field(
        default=None,
        description="Bhoonidhi API key (set via environment variable)"
    )
    
    # UI Settings
    streamlit_port: int = Field(
        default=8501,
        description="Streamlit server port"
    )
    map_default_zoom: int = Field(
        default=10,
        description="Default zoom level for maps"
    )
    map_default_center: List[float] = Field(
        default=[20.5937, 78.9629],  # India center
        description="Default map center [lat, lon]"
    )
    
    # Security Settings
    enable_code_sandboxing: bool = Field(
        default=True,
        description="Enable secure code sandboxing"
    )
    allowed_imports: List[str] = Field(
        default=[
            "geopandas", "pandas", "numpy", "shapely", "rasterio",
            "fiona", "pyproj", "json", "math", "datetime", "os.path",
            "whitebox", "rasterstats", "osmnx"
        ],
        description="Allowed Python imports in sandbox"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses LRU cache for efficiency in repeated calls.
    
    Returns:
        Settings: Validated configuration settings
    """
    return Settings()


def create_directories(settings: Optional[Settings] = None) -> None:
    """
    Create necessary directories for the application.
    
    Args:
        settings: Optional settings instance, uses default if not provided
    """
    if settings is None:
        settings = get_settings()
    
    directories = [
        settings.vector_store_path,
        settings.docs_path,
        settings.osm_cache_dir,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
