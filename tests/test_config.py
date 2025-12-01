"""Tests for configuration module."""

import os
from pathlib import Path

import pytest

from geo_reason.core.config import Settings, get_settings, create_directories


class TestSettings:
    """Test cases for Settings class."""
    
    def test_default_settings(self):
        """Test that default settings are properly initialized."""
        settings = Settings()
        
        assert settings.app_name == "Geo-Reason"
        assert settings.app_version == "1.0.0"
        assert settings.debug is False
        assert settings.log_level == "INFO"
    
    def test_model_settings(self):
        """Test model-related settings."""
        settings = Settings()
        
        assert "llama" in settings.model_name.lower() or "meta" in settings.model_name.lower()
        assert settings.model_max_tokens == 4096
        assert settings.model_temperature == 0.1
        assert settings.use_quantization is True
    
    def test_sandbox_settings(self):
        """Test sandbox-related settings."""
        settings = Settings()
        
        assert settings.sandbox_timeout == 300
        assert settings.max_retries == 3
        assert settings.enable_code_sandboxing is True
    
    def test_allowed_imports(self):
        """Test that allowed imports are configured."""
        settings = Settings()
        
        assert "geopandas" in settings.allowed_imports
        assert "pandas" in settings.allowed_imports
        assert "numpy" in settings.allowed_imports
        assert "shapely" in settings.allowed_imports
    
    def test_environment_override(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        
        # Clear cache to get fresh settings
        get_settings.cache_clear()
        settings = Settings()
        
        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        
        # Reset cache
        get_settings.cache_clear()
    
    def test_get_settings_caching(self):
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2


class TestCreateDirectories:
    """Test cases for create_directories function."""
    
    def test_create_directories(self, tmp_path):
        """Test that directories are created properly."""
        settings = Settings(
            vector_store_path=tmp_path / "vector_store",
            docs_path=tmp_path / "docs",
            osm_cache_dir=tmp_path / "osm_cache"
        )
        
        create_directories(settings)
        
        assert settings.vector_store_path.exists()
        assert settings.docs_path.exists()
        assert settings.osm_cache_dir.exists()
