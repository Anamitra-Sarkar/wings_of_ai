"""
Logging configuration for Geo-Reason system.

Provides structured logging with context support for enterprise applications.
"""

import sys
from functools import lru_cache
from typing import Optional

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    include_context: bool = True
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_format: Use JSON formatting for logs (recommended for production)
        include_context: Include contextual information in logs
    """
    # Remove default handler
    logger.remove()
    
    # Console format with color support
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    if include_context:
        console_format += " | <dim>{extra}</dim>"
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler if specified
    if log_file:
        if json_format:
            logger.add(
                log_file,
                format="{message}",
                level=level,
                serialize=True,
                rotation="100 MB",
                retention="30 days",
                compression="gz"
            )
        else:
            logger.add(
                log_file,
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                    "{name}:{function}:{line} | {message}"
                ),
                level=level,
                rotation="100 MB",
                retention="30 days",
                compression="gz"
            )


@lru_cache()
def get_logger(name: str = "geo_reason"):
    """
    Get a contextualized logger instance.
    
    Args:
        name: Logger name/module identifier
        
    Returns:
        Configured logger instance
    """
    return logger.bind(module=name)


class LogContext:
    """
    Context manager for adding contextual information to logs.
    
    Usage:
        with LogContext(query_id="abc123", user="admin"):
            logger.info("Processing query")
    """
    
    def __init__(self, **context):
        """Initialize with context key-value pairs."""
        self.context = context
        self._token = None
    
    def __enter__(self):
        """Enter context and bind values."""
        self._token = logger.contextualize(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and unbind values."""
        if self._token:
            self._token.__exit__(exc_type, exc_val, exc_tb)
        return False
