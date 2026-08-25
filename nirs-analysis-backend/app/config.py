"""
Configuration settings for the NIRS analysis application.
"""
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class Config:
    """Base configuration class for the application."""
    APP_NAME = "NIRS Analysis Backend"
    DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
    SECRET_KEY = os.getenv("SECRET_KEY", "nirs_secret_development_key")

    # File upload settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64 MB

    # Data folders
    PROCESSED_DATA_FOLDER = os.path.join(BASE_DIR, 'data', 'processed')
    TEMP_DATA_FOLDER = os.path.join(BASE_DIR, 'data', 'temp')

    # Logging
    LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
