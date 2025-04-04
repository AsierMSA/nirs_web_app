# File: /nirs-analysis-backend/nirs-analysis-backend/app/utils/file_handlers.py

import os
from flask import request, abort

def save_uploaded_file(upload_folder):
    """
    Save the uploaded NIRS file to the specified upload folder.

    Parameters:
    -----------
    upload_folder : str
        The folder where uploaded files will be stored.

    Returns:
    --------
    str
        The path to the saved file.
    """
    if 'file' not in request.files:
        abort(400, 'No file part in the request')
    
    file = request.files['file']
    
    if file.filename == '':
        abort(400, 'No selected file')
    
    # Ensure the upload folder exists
    os.makedirs(upload_folder, exist_ok=True)
    
    # Save the file
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    
    return file_path

def get_file_path(filename, base_folder):
    """
    Construct the full file path for a given filename in the specified base folder.

    Parameters:
    -----------
    filename : str
        The name of the file.
    base_folder : str
        The base folder where the file is located.

    Returns:
    --------
    str
        The full path to the file.
    """
    return os.path.join(base_folder, filename)

def delete_file(file_path):
    """
    Delete a file at the specified path.

    Parameters:
    -----------
    file_path : str
        The path to the file to be deleted.

    Returns:
    --------
    bool
        True if the file was deleted successfully, False otherwise.
    """
    try:
        os.remove(file_path)
        return True
    except OSError:
        return False