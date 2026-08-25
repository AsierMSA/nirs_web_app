"""
API routes for NIRS data analysis, file management, and ML pipelines.
"""
import os
import traceback
import logging
from flask import Blueprint, request, jsonify, current_app, make_response
from werkzeug.utils import secure_filename

from app.core.nirs_processor import analyze_nirs_file, load_nirs_data, extract_features_from_raw
from app.core.nirs_ml import validate_against_temporal_bias
from app.utils.response_formatter import convert_numpy_types

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)

ALLOWED_EXTENSIONS = {'fif', 'gz'}

def _is_allowed_file(filename):
    lower = filename.lower()
    return lower.endswith('.fif') or lower.endswith('.fif.gz') or lower.endswith('.gz')

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify backend status."""
    try:
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        return jsonify({
            'status': 'healthy',
            'app_name': current_app.config.get('APP_NAME', 'NIRS Analysis Backend'),
            'upload_folder_exists': os.path.exists(upload_folder),
            'max_file_size_mb': current_app.config.get('MAX_CONTENT_LENGTH', 67108864) // (1024 * 1024)
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload requests for NIRS data files (.fif, .fif.gz)."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not _is_allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload .fif or .fif.gz files'}), 400

        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)

        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)

        # Handle duplicate filenames safely
        if os.path.exists(file_path):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(file_path):
                new_filename = f"{base}_{counter}{ext}"
                file_path = os.path.join(upload_folder, new_filename)
                counter += 1
            filename = os.path.basename(file_path)

        file.save(file_path)

        # Validate NIRS data
        try:
            test_data = load_nirs_data(file_path)
            if test_data is None:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'error': 'Invalid NIRS file format or corrupt signal'}), 400

            file_info = {
                'id': filename,
                'name': filename,
                'size': os.path.getsize(file_path),
                'channels': len(test_data.ch_names),
                'duration': float(test_data.times[-1]) if len(test_data.times) > 0 else 0.0,
                'sampling_freq': float(test_data.info['sfreq']),
                'annotations': len(test_data.annotations)
            }
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'Failed to validate NIRS file: {str(e)}'}), 400

        return jsonify({
            'message': 'File uploaded and validated successfully',
            'file': file_info,
            'file_id': filename,
            'filename': filename
        }), 200

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@api_bp.route('/files', methods=['GET'])
def get_files():
    """List all available NIRS files with metadata."""
    try:
        upload_folder = current_app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            return jsonify({'files': []}), 200

        files = []
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path) and _is_allowed_file(filename):
                try:
                    file_info = {
                        'id': filename,
                        'name': filename,
                        'size': os.path.getsize(file_path)
                    }
                    try:
                        raw_data = load_nirs_data(file_path)
                        if raw_data is not None:
                            file_info.update({
                                'channels': len(raw_data.ch_names),
                                'duration': float(raw_data.times[-1]) if len(raw_data.times) > 0 else 0.0,
                                'sampling_freq': float(raw_data.info['sfreq']),
                                'annotations': len(raw_data.annotations)
                            })
                    except Exception:
                        pass
                    files.append(file_info)
                except Exception as e:
                    logger.warning(f"Error processing file {filename}: {e}")
                    continue

        return jsonify({'files': files}), 200
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        return jsonify({'error': f'Failed to list files: {str(e)}'}), 500

@api_bp.route('/available_activities', methods=['GET'])
def get_available_activities():
    """Extract distinct cognitive/motor activity annotations from a NIRS file."""
    try:
        file_id = request.args.get('file_id')
        if not file_id:
            return jsonify({'error': 'No file ID provided'}), 400

        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(file_id))
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {file_id}'}), 404

        raw_data = load_nirs_data(file_path)
        if raw_data is None:
            return jsonify({'error': 'Failed to load NIRS data'}), 400

        activity_info = {}
        for annot in raw_data.annotations:
            activity = str(annot['description'])
            if activity not in activity_info:
                activity_info[activity] = {
                    'name': activity,
                    'count': 0,
                    'total_duration': 0.0,
                    'first_occurrence': float(annot['onset']),
                    'last_occurrence': float(annot['onset'])
                }
            activity_info[activity]['count'] += 1
            activity_info[activity]['total_duration'] += float(annot['duration'])
            activity_info[activity]['last_occurrence'] = max(
                activity_info[activity]['last_occurrence'],
                float(annot['onset'])
            )

        activities = list(activity_info.values())
        activities.sort(key=lambda x: x['first_occurrence'])

        return jsonify({
            'activities': [a['name'] for a in activities],
            'activity_details': activities,
            'file_id': file_id,
            'total_duration': float(raw_data.times[-1]) if len(raw_data.times) > 0 else 0.0,
            'total_annotations': len(raw_data.annotations)
        }), 200
    except Exception as e:
        logger.error(f"Failed to get activities: {e}")
        return jsonify({'error': f'Failed to get activities: {str(e)}'}), 500

@api_bp.route('/analyze', methods=['POST'])
def analyze_data():
    """Run full NIRS signal processing, feature extraction, and ML classification pipeline."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        file_id = data.get('file_id')
        activities = data.get('activities', [])
        annotation_map = data.get('annotation_map')

        if not file_id:
            return jsonify({'error': 'No file ID provided'}), 400
        if not activities:
            return jsonify({'error': 'At least one activity must be provided'}), 400

        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(file_id))
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {file_id}'}), 404

        analysis_result = analyze_nirs_file(file_path, activities, annotation_map)
        if 'error' in analysis_result:
            return jsonify({'error': analysis_result['error']}), 400

        converted_result = convert_numpy_types(analysis_result)
        return jsonify({
            'message': 'Analysis completed successfully',
            'file_id': file_id,
            'activities': activities,
            **converted_result
        }), 200

    except Exception as e:
        logger.error(f"Analysis failed: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@api_bp.route('/temporal_validation', methods=['POST', 'OPTIONS'])
def temporal_validation():
    """Evaluate classifier performance across chronologically split folds to detect temporal bias."""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        file_id = data.get('file_id')
        activities = data.get('activities', [])

        if not file_id or not activities:
            return jsonify({'error': 'Missing file_id or activities'}), 400

        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], secure_filename(file_id))
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        raw_data = load_nirs_data(file_path)
        if raw_data is None:
            return jsonify({'error': 'Failed to load NIRS data'}), 400

        features_result = extract_features_from_raw(raw_data, activities)
        if 'error' in features_result:
            return jsonify({'error': features_result['error']}), 400

        temp_val = validate_against_temporal_bias(
            features_result['X_features'],
            features_result['labels'],
            features_result['feature_names']
        )

        return jsonify({
            'temporal_validation': convert_numpy_types(temp_val)
        }), 200

    except Exception as e:
        logger.error(f"Temporal validation failed: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Temporal validation failed: {str(e)}'}), 500
