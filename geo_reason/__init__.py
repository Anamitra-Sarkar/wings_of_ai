"""
Geo-Reason: An Enterprise GeoAI Platform

A system that uses LLMs to autonomously plan and execute complex
geospatial analysis tasks based on natural language queries.
"""

__version__ = "1.0.0"
__author__ = "Geo-Reason Team"

from geo_reason.core.config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
