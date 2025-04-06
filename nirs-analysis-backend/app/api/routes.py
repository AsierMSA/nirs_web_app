from flask import Blueprint, request, jsonify, current_app
from app.api.validators import validate_file_upload
from app.core.nirs_processor import analyze_nirs_data, analyze_nirs_file
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
    if not validate_file_upload(file):
        return jsonify({'error': 'Invalid file format. Please upload a .fif or .fif.gz file.'}), 400
    
    # Save the uploaded file
    file_path = save_uploaded_file(file)
    
    # Analyze the NIRS data
    try:
        results = analyze_nirs_data(file_path, activity_names=["Finger Sequencing", "Simple Tapping", "Motor Imagery", "Bimanual Coordination", "Working Memory", "Rest"])
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        "filename": "example.fif.gz",
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
        
        filename = data.get('filename')
        activities = data.get('activities', [])
        
        if not filename:
            return jsonify({
                'status': 'error',
                'message': 'Filename is required'
            }), 400
            
        if not activities:
            return jsonify({
                'status': 'error',
                'message': 'At least one activity must be selected'
            }), 400
        
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(file_path):
            return jsonify({
                'status': 'error',
                'message': 'File not found'
            }), 404
        
        # Perform the NIRS analysis
        results = analyze_nirs_file(file_path, activities)
        
        if 'error' in results:
            return jsonify({
                'status': 'error',
                'message': results['error'],
                'details': results.get('traceback', '')
            }), 400
        
        # Add analysis metadata
        results['filename'] = filename
        results['activities'] = activities
        
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Error in analysis: {str(e)}'
        }), 500