"""
Main entry point for the NIRS analysis backend application.
"""

from flask import Flask, make_response, request, jsonify
from flask_cors import CORS
import os
import traceback
from werkzeug.utils import secure_filename  # Add this import for secure_filename

# Fix imports - remove the folder with dash from import path
from app.core.nirs_processor import analyze_nirs_file, load_nirs_data
from app.core.plotting import generate_plots_for_api

def create_app():
    """
    Create and configure the Flask application.
    """
    app = Flask(__name__)
    
    # Configure CORS to allow requests from frontend
    CORS(app)
    
    # Configure app
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size
    
    # Create upload directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    @app.route('/')
    def index():
        return jsonify({'status': 'NIRS Analysis API is running'})
    
    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        """
        Handle file upload requests for NIRS data files.
        """
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Check file extension (allow .fif, .fif.gz)
        allowed_extensions = {'fif', 'gz'}
        if not ('.' in file.filename and 
                (file.filename.rsplit('.', 1)[1].lower() in allowed_extensions or
                 file.filename.endswith('.fif.gz'))):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'file_id': filename  # Using filename as file_id for simplicity
        }), 200

    @app.route('/api/files', methods=['GET'])
    def get_files():
        """
        Get a list of available NIRS files.
        """
        upload_folder = app.config['UPLOAD_FOLDER']
        
        if not os.path.exists(upload_folder):
            return jsonify({'files': []}), 200
        
        # Get all files in the upload folder
        files = []
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path) and (filename.endswith('.fif') or filename.endswith('.fif.gz')):
                files.append({
                    'file_id': filename,
                    'filename': filename,
                    'size': os.path.getsize(file_path)
                })
        
        return jsonify({'files': files}), 200

    @app.route('/api/available_activities', methods=['GET'])
    def get_available_activities():
        """
        Get a list of available activities in a NIRS file.
        """
        file_id = request.args.get('file_id')
        
        if not file_id:
            return jsonify({'error': 'No file ID provided'}), 400
        
        # Get the file path
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Use directly load_nirs_data that was imported above
        raw_data = load_nirs_data(file_path)
        
        if raw_data is None:
            return jsonify({'error': 'Failed to load NIRS data'}), 400
        
        # Get unique activity names from annotations
        activities = []
        for annot in raw_data.annotations:
            activity = annot['description']
            if activity not in activities:
                activities.append(activity)
        
        return jsonify({
            'activities': activities,
            'file_id': file_id
        }), 200
    
    @app.route('/api/analyze', methods=['POST'])
    def analyze_data():
        """
        Analyze NIRS data and return results with plots.
        """
        try:
            # Get request parameters
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            file_id = data.get('file_id')
            activities = data.get('activities', [])
            
            if not file_id:
                return jsonify({'error': 'No file ID provided'}), 400
            
            if not activities:
                return jsonify({'error': 'No activities provided'}), 400
            
            # Get the file path
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
            
            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404
            
            # Analyze the file
            analysis_result = analyze_nirs_file(file_path, activities)
            
            if 'error' in analysis_result:
                return jsonify({'error': analysis_result['error']}), 400
            
            # Generate plots
            plots = generate_plots_for_api(analysis_result, activities)
            
            # Prepare the response
            response = {
                'message': 'Analysis completed successfully',
                'features': analysis_result.get('features', {}),
                'plots': plots,
                'file_id': file_id,
                'activities': activities
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            print(traceback.format_exc())
            return jsonify({'error': f'Server error: {str(e)}'}), 500
    @app.route('/api/temporal_validation', methods=['POST', 'OPTIONS'])
    def temporal_validation():
            """
            Analyze NIRS data with temporal validation to test for bias.
            """
            # Handle preflight OPTIONS request for CORS
            if request.method == 'OPTIONS':
                return _build_cors_preflight_response()
                
            try:
                # Get request parameters
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                file_id = data.get('file_id')
                activities = data.get('activities', [])
                
                if not file_id:
                    return jsonify({'error': 'No file ID provided'}), 400
                
                if not activities:
                    return jsonify({'error': 'No activities provided'}), 400
                
                # Get the file path
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
                
                if not os.path.exists(file_path):
                    return jsonify({'error': 'File not found'}), 404
                
                # Load the data
                raw_data = load_nirs_data(file_path)
                if raw_data is None:
                    return jsonify({'error': 'Failed to load NIRS data'}), 400
                    
                # Extract features and run temporal validation
                from app.core.nirs_ml import validate_against_temporal_bias
                from app.core.nirs_processor import extract_features_from_raw
                
                # Extract features from raw data
                features_result = extract_features_from_raw(raw_data, activities)
                
                # Run temporal validation
                temporal_validation = validate_against_temporal_bias(
                    features_result['X_features'],
                    features_result['labels'],
                    features_result['feature_names']
                )
                converted_result = {
                    'temporal_validation': convert_numpy_types(temporal_validation)
                }
                # Return results
                return jsonify(converted_result), 200
                
            except Exception as e:
                print(traceback.format_exc())
                return jsonify({'error': f'Server error: {str(e)}'}), 500

        # Helper function for CORS preflight responses
    def _build_cors_preflight_response():
            response = make_response()
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add("Access-Control-Allow-Headers", "Content-Type")
            response.headers.add("Access-Control-Allow-Methods", "POST")
            return response
    def convert_numpy_types(obj):
        """Convert NumPy types to standard Python types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj  
    return app
# Add this after the '/api/analyze' endpoint function

   
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)