# Configuration settings for the NIRS analysis application

import os

class Config:
    """Base configuration class for the application."""
    
    # General settings
    APP_NAME = "NIRS Analysis Backend"
    DEBUG = os.getenv("DEBUG", "False") == "True"  # Set to True for development
    SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")  # Secret key for session management

    # File upload settings
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'data', 'uploads')  # Directory for uploaded files
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Limit upload size to 16 MB

    # Data processing settings
    PROCESSED_DATA_FOLDER = os.path.join(os.getcwd(), 'data', 'processed')  # Directory for processed data
    TEMP_DATA_FOLDER = os.path.join(os.getcwd(), 'data', 'temp')  # Directory for temporary files

    # Logging settings
    LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")  # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    # Add any additional configuration settings as needed

# You can create different configuration classes for development, testing, and production by subclassing Config.