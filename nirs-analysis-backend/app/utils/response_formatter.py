# File: /nirs-analysis-backend/nirs-analysis-backend/app/utils/response_formatter.py

def format_success_response(data, message="Operation successful"):
    """
    Formats a successful response for the API.

    Parameters:
    data (dict): The data to include in the response.
    message (str): A success message to include in the response.

    Returns:
    dict: A formatted response dictionary.
    """
    return {
        "status": "success",
        "message": message,
        "data": data
    }

def format_error_response(error_message, status_code=400):
    """
    Formats an error response for the API.

    Parameters:
    error_message (str): The error message to include in the response.
    status_code (int): The HTTP status code for the error.

    Returns:
    dict: A formatted error response dictionary.
    """
    return {
        "status": "error",
        "message": error_message,
        "code": status_code
    }

def format_analysis_result(result):
    """
    Formats the analysis result for the API response.

    Parameters:
    result (dict): The analysis result containing accuracy and feature importance.

    Returns:
    dict: A formatted analysis result dictionary.
    """
    return {
        "accuracy": result.get("accuracy"),
        "feature_importance": result.get("region_importance"),
        "features": result.get("features")
    }