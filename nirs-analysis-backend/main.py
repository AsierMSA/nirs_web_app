"""
Main entry point for the NIRS analysis backend application.
"""

from flask import Flask, make_response, request, jsonify
from flask_cors import CORS
import os
import traceback
import numpy as np
from werkzeug.utils import secure_filename

# Fix imports - remove the folder with dash from import path
from app.core.nirs_processor import analyze_nirs_file, load_nirs_data

def create_app():
    """
    Create and configure the Flask application.
    """
    app = Flask(__name__)
    
    # Configure CORS to allow requests from frontend
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Configure app
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # Increased to 64MB for larger files
    
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    @app.route('/')
    def index():
        return jsonify({
            'status': 'NIRS Analysis API is running',
            'version': '2.0',
            'features': [
                'File upload and management',
                'Activity detection',
                'Advanced signal analysis',
                'Machine learning classification',
                'Temporal validation',
                'Brain connectivity analysis',
                'Spectral analysis',
                'Signal quality assessment'
            ]
        })
    
    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        """Handle file upload requests for NIRS data files."""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part in request'}), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Enhanced file validation
            allowed_extensions = {'fif', 'gz'}
            filename_lower = file.filename.lower()
            
            if not (filename_lower.endswith('.fif') or 
                   filename_lower.endswith('.fif.gz') or
                   filename_lower.endswith('.gz')):
                return jsonify({'error': 'Invalid file type. Please upload .fif or .fif.gz files'}), 400
            
            # Secure filename and save
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Check if file already exists
            if os.path.exists(file_path):
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(file_path):
                    new_filename = f"{base}_{counter}{ext}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
                    counter += 1
                filename = os.path.basename(file_path)
            
            file.save(file_path)
            
            # Verify file can be loaded
            try:
                test_data = load_nirs_data(file_path)
                if test_data is None:
                    os.remove(file_path)  # Clean up invalid file
                    return jsonify({'error': 'Invalid NIRS file format'}), 400
                    
                file_info = {
                    'id': filename,
                    'name': filename,
                    'size': os.path.getsize(file_path),
                    'channels': len(test_data.ch_names),
                    'duration': test_data.times[-1],
                    'sampling_freq': test_data.info['sfreq'],
                    'annotations': len(test_data.annotations)
                }
                
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'error': f'Failed to validate NIRS file: {str(e)}'}), 400
            
            return jsonify({
                'message': 'File uploaded and validated successfully',
                'file': file_info
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Upload failed: {str(e)}'}), 500

    @app.route('/api/files', methods=['GET'])
    def get_files():
        """Get a list of available NIRS files with metadata."""
        try:
            upload_folder = app.config['UPLOAD_FOLDER']
            
            if not os.path.exists(upload_folder):
                return jsonify({'files': []}), 200
            
            files = []
            for filename in os.listdir(upload_folder):
                file_path = os.path.join(upload_folder, filename)
                
                if (os.path.isfile(file_path) and 
                    (filename.endswith('.fif') or filename.endswith('.fif.gz'))):
                    
                    try:
                        # Get basic file info
                        file_info = {
                            'id': filename,
                            'name': filename,
                            'size': os.path.getsize(file_path)
                        }
                        
                        # Try to get NIRS metadata
                        try:
                            raw_data = load_nirs_data(file_path)
                            if raw_data is not None:
                                file_info.update({
                                    'channels': len(raw_data.ch_names),
                                    'duration': float(raw_data.times[-1]),
                                    'sampling_freq': float(raw_data.info['sfreq']),
                                    'annotations': len(raw_data.annotations)
                                })
                        except:
                            # If metadata extraction fails, still include the file
                            pass
                            
                        files.append(file_info)
                        
                    except Exception as e:
                        print(f"Error processing file {filename}: {e}")
                        continue
            
            return jsonify({'files': files}), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to list files: {str(e)}'}), 500

    @app.route('/api/available_activities', methods=['GET'])
    def get_available_activities():
        """Get available activities in a NIRS file with metadata."""
        try:
            file_id = request.args.get('file_id')
            
            if not file_id:
                return jsonify({'error': 'No file ID provided'}), 400
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
            
            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404
            
            raw_data = load_nirs_data(file_path)
            
            if raw_data is None:
                return jsonify({'error': 'Failed to load NIRS data'}), 400
            
            # Extract activities with counts and timing info
            activity_info = {}
            for annot in raw_data.annotations:
                activity = str(annot['description'])
                if activity not in activity_info:
                    activity_info[activity] = {
                        'name': activity,
                        'count': 0,
                        'total_duration': 0,
                        'first_occurrence': float(annot['onset']),
                        'last_occurrence': float(annot['onset'])
                    }
                
                activity_info[activity]['count'] += 1
                activity_info[activity]['total_duration'] += float(annot['duration'])
                activity_info[activity]['last_occurrence'] = max(
                    activity_info[activity]['last_occurrence'], 
                    float(annot['onset'])
                )
            
            # Convert to list and sort by first occurrence
            activities = list(activity_info.values())
            activities.sort(key=lambda x: x['first_occurrence'])
            
            return jsonify({
                'activities': [a['name'] for a in activities],
                'activity_details': activities,
                'file_id': file_id,
                'total_duration': float(raw_data.times[-1]),
                'total_annotations': len(raw_data.annotations)
            }), 200
            
        except Exception as e:
            return jsonify({'error': f'Failed to get activities: {str(e)}'}), 500

    @app.route('/api/analyze', methods=['POST'])
    def analyze_data():
        """Analyze NIRS data with advanced visualizations."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            file_id = data.get('file_id')
            activities = data.get('activities', [])
            annotation_map = data.get('annotation_map')  # Optional annotation mapping
            
            if not file_id:
                return jsonify({'error': 'No file ID provided'}), 400
            
            if not activities:
                return jsonify({'error': 'No activities provided'}), 400
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
            
            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404
            
            print(f"Starting analysis for {file_id} with activities: {activities}")
            
            # Run the enhanced analysis
            analysis_result = analyze_nirs_file(file_path, activities, annotation_map)
            
            if 'error' in analysis_result:
                return jsonify({'error': analysis_result['error']}), 400
            
            # Convert numpy types for JSON serialization
            converted_result = convert_numpy_types(analysis_result)
            
            print(f"Analysis completed successfully for {file_id}")
            print(f"Available plots: {list(converted_result.get('plots', {}).keys())}")
            
            return jsonify({
                'message': 'Analysis completed successfully',
                'file_id': file_id,
                'activities': activities,
                **converted_result  # Spread all analysis results
            }), 200
            
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
        
    @app.route('/api/temporal_validation', methods=['POST', 'OPTIONS'])
    def temporal_validation():
        """Analyze NIRS data with temporal validation to test for bias."""
        if request.method == 'OPTIONS':
            return _build_cors_preflight_response()
            
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            file_id = data.get('file_id')
            activities = data.get('activities', [])
            
            if not file_id:
                return jsonify({'error': 'No file ID provided'}), 400
            
            if not activities:
                return jsonify({'error': 'No activities provided'}), 400
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
            
            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404
            
            raw_data = load_nirs_data(file_path)
            if raw_data is None:
                return jsonify({'error': 'Failed to load NIRS data'}), 400
                
            # Import ML functions
            from app.core.nirs_ml import validate_against_temporal_bias
            from app.core.nirs_processor import extract_features_from_raw
            
            # Extract features
            features_result = extract_features_from_raw(raw_data, activities)
            
            if 'error' in features_result:
                return jsonify({'error': features_result['error']}), 400
            
            # Run temporal validation
            temporal_validation = validate_against_temporal_bias(
                features_result['X_features'],
                features_result['labels'],
                features_result['feature_names']
            )
            
            converted_result = {
                'temporal_validation': convert_numpy_types(temporal_validation)
            }
            
            return jsonify(converted_result), 200
            
        except Exception as e:
            print(f"Temporal validation error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'Temporal validation failed: {str(e)}'}), 500

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        try:
            # Test basic functionality
            upload_folder_exists = os.path.exists(app.config['UPLOAD_FOLDER'])
            
            # Test imports
            from app.core.nirs_processor import load_nirs_data
            from app.core.nirs_ml import validate_against_temporal_bias
            
            return jsonify({
                'status': 'healthy',
                'upload_folder_exists': upload_folder_exists,
                'upload_folder_path': app.config['UPLOAD_FOLDER'],
                'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
            }), 200
            
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'error': str(e)
            }), 500

    @app.after_request
    def add_headers(response):
        """Add headers to prevent caching and enable CORS."""
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        # Additional CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        
        return response

    def _build_cors_preflight_response():
        """Helper function for CORS preflight responses."""
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response

    def convert_numpy_types(obj):
        """Convert NumPy types to standard Python types for JSON serialization."""
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
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    # Error handlers
    @app.errorhandler(413)
    def file_too_large(error):
        return jsonify({'error': 'File too large. Maximum size is 64MB.'}), 413
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    print("Starting NIRS Analysis Backend...")
    print("Available endpoints:")
    print("  GET  / - API status")
    print("  GET  /api/health - Health check")
    print("  POST /api/upload - Upload NIRS file")
    print("  GET  /api/files - List uploaded files")
    print("  GET  /api/available_activities - Get file activities")
    print("  POST /api/analyze - Analyze NIRS data")
    print("  POST /api/temporal_validation - Temporal validation")
    app.run(debug=True, host='0.0.0.0', port=5000)