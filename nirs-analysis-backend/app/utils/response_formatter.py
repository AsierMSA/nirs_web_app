"""
Utilities for formatting API responses and JSON serialization.
"""
import numpy as np

def convert_numpy_types(obj):
    """Convert NumPy types to standard Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def format_success_response(data, message="Operation successful"):
    return {
        "status": "success",
        "message": message,
        "data": convert_numpy_types(data)
    }

def format_error_response(error_message, status_code=400):
    return {
        "status": "error",
        "message": error_message,
        "code": status_code
    }
