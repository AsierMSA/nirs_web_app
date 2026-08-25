"""
Unit tests for data formatting and core NIRS processor utilities.
"""
import numpy as np
import pytest
from app.utils.response_formatter import convert_numpy_types, format_success_response, format_error_response
from app.core.nirs_processor import load_nirs_data, analyze_nirs_file

def test_convert_numpy_types():
    """Verify numpy scalar and array types are converted to JSON serializable Python types."""
    data = {
        'int_val': np.int64(42),
        'float_val': np.float32(3.14159),
        'bool_val': np.bool_(True),
        'array_val': np.array([1, 2, 3]),
        'nested': {
            'matrix': np.ones((2, 2))
        }
    }
    converted = convert_numpy_types(data)
    assert isinstance(converted['int_val'], int)
    assert isinstance(converted['float_val'], float)
    assert isinstance(converted['bool_val'], bool)
    assert isinstance(converted['array_val'], list)
    assert isinstance(converted['nested']['matrix'], list)

def test_format_responses():
    """Verify standard response structures."""
    success = format_success_response({"accuracy": 0.85}, "Done")
    assert success['status'] == 'success'
    assert success['data']['accuracy'] == 0.85

    error = format_error_response("Something went wrong", 400)
    assert error['status'] == 'error'
    assert error['code'] == 400

def test_load_nirs_data_invalid_path():
    """Verify loading non-existent file returns None without throwing unhandled exceptions."""
    result = load_nirs_data("non_existent_file_path.fif.gz")
    assert result is None

def test_analyze_nirs_file_missing():
    """Verify analysis on missing file returns a clear error dictionary."""
    result = analyze_nirs_file("missing_file.fif.gz", ["Activity A"])
    assert 'error' in result
