"""
Logging configuration for FinReport RAG
Provides both console and file logging with debug mode support
"""
import logging
import os
from datetime import datetime
from pathlib import Path

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Configuration
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"

# Create logger
logger = logging.getLogger("finreport_rag")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

# Remove existing handlers
logger.handlers.clear()

# Console handler (only in debug mode)
if DEBUG_MODE:
    import sys
    # Create a UTF-8 compatible stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

# File handler (always enabled if flag is set)
if ENABLE_FILE_LOGGING:
    log_filename = LOGS_DIR / f"finreport_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

# Prevent propagation to root logger
logger.propagate = False


def log_info(message: str):
    """Log info message"""
    logger.info(message)


def log_error(message: str, exc_info=False):
    """Log error message"""
    logger.error(message, exc_info=exc_info)


def log_debug(message: str):
    """Log debug message"""
    logger.debug(message)


def log_warning(message: str):
    """Log warning message"""
    logger.warning(message)


def log_exception(message: str, exception: Exception):
    """Log exception with traceback"""
    logger.exception(f"{message}: {str(exception)}")
