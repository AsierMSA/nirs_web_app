from flask import Blueprint, request, jsonify, current_app
from app.api.validators import validate_file_upload
from app.core.nirs_processor import analyze_nirs_file
from app.utils.file_handlers import save_uploaded_file
import os
import logging
from werkzeug.utils import secure_filename

# Configure logger
logger = logging.getLogger(__name__)
# Create a blueprint for the API routes
api_bp = Blueprint('api', __name__)

@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """
    Endpoint to upload NIRS files for analysis.
    Expects a file in the request.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Validate the uploaded file
    # Assuming validate_file_upload checks extension and maybe content
    # if not validate_file_upload(file): # This function seems incomplete/not used
    #     return jsonify({'error': 'Invalid file format. Please upload a .fif or .fif.gz file.'}), 400
    
    # Basic filename check
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Check file extension (allow .fif, .fif.gz)
    allowed_extensions = {'fif', 'gz'}
    if not ('.' in file.filename and 
            (file.filename.rsplit('.', 1)[1].lower() in allowed_extensions or
             file.filename.endswith('.fif.gz'))):
        return jsonify({'error': 'Invalid file type. Please upload a .fif or .fif.gz file.'}), 400

    # Save the uploaded file using secure_filename
    filename = secure_filename(file.filename)
    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True) # Ensure folder exists
    file_path = os.path.join(upload_folder, filename)
    
    try:
        file.save(file_path)
        # Return success response with filename/ID
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'file_id': filename # Using filename as ID for simplicity
        }), 200
    except Exception as e:
        logger.error(f"Error saving file {filename}: {e}")
        return jsonify({'error': f'Could not save file: {str(e)}'}), 500


@api_bp.route('/results', methods=['GET'])
def get_results():
    """
    Endpoint to retrieve analysis results.
    This is a placeholder for future implementation.
    """
    return jsonify({'message': 'Results retrieval is not yet implemented.'}), 501

@api_bp.route('/analyze', methods=['POST'])
def analyze_nirs():
    """
    Analyze NIRS data with selected activities.
    
    JSON Body:
    ---------
    {
        "file_id": "example.fif.gz", # Changed from filename to file_id
        "activities": ["Finger Sequencing", "Simple Tapping", ...]
    }
    
    Returns:
    --------
    JSON response with analysis results including plots.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        file_id = data.get('file_id') # Use file_id
        activities = data.get('activities', [])
        
        if not file_id:
            return jsonify({
                'status': 'error',
                'message': 'File ID is required' # Changed from Filename
            }), 400
            
        if not activities:
            return jsonify({
                'status': 'error',
                'message': 'At least one activity must be selected'
            }), 400
        
        # Construct path using file_id and secure_filename just in case
        filename = secure_filename(file_id) 
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                'status': 'error',
                'message': f'File not found: {filename}' # More informative error
            }), 404
        
        # Perform the NIRS analysis using analyze_nirs_file
        results = analyze_nirs_file(file_path, activities)
        
        if 'error' in results:
            return jsonify({
                'status': 'error',
                'message': results['error'],
                'details': results.get('traceback', '')
            }), 400 # Or 500 if it's a server-side processing error
        
        # Add analysis metadata
        results['file_id'] = file_id # Use file_id
        results['activities'] = activities
        
        # Return the results directly (main.py structure)
        return jsonify(results) # Removed 'status' and 'results' nesting
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error', # Keep status here for general errors
            'message': f'Error in analysis: {str(e)}'
        }), 500