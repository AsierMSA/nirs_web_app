# This file contains validation functions for incoming requests, ensuring that the data meets the required formats and constraints.

from pydantic import BaseModel, FilePath, validator
from typing import List

class NIRSFileUpload(BaseModel):
    """
    Model for validating NIRS file uploads.
    """
    files: List[FilePath]  # List of file paths for uploaded NIRS files

    @validator('files', each_item=True)
    def validate_file_extension(cls, file_path):
        """
        Validate that the uploaded file has a .fif or .fif.gz extension.
        """
        if not (file_path.endswith('.fif') or file_path.endswith('.fif.gz')):
            raise ValueError('File must be a .fif or .fif.gz file')
        return file_path

class AnalysisRequest(BaseModel):
    """
    Model for validating analysis requests.
    """
    activity_names: List[str]  # List of activity names to analyze

    @validator('activity_names')
    def validate_activity_names(cls, activities):
        """
        Validate that activity names are not empty and are strings.
        """
        if not activities:
            raise ValueError('Activity names cannot be empty')
        for activity in activities:
            if not isinstance(activity, str) or not activity.strip():
                raise ValueError('Each activity name must be a non-empty string')
        return activities

# Additional validators can be added here as needed for other request types.