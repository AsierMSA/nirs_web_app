"""
Main entry point for the NIRS analysis backend application.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback
from werkzeug.utils import secure_filename  # Add this import for secure_filename

# Fix imports - remove the folder with dash from import path
from app.core.analyzer import analyze_nirs_file, load_nirs_data
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
            
            # Return analysis results and plots
            return jsonify({
                'message': 'Analysis completed successfully',
                'plots': plots,
                'file_id': file_id,
                'activities': activities
            }), 200
            
        except Exception as e:
            print(traceback.format_exc())
            return jsonify({'error': f'Server error: {str(e)}'}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)