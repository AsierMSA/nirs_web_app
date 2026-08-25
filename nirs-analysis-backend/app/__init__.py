"""
Flask application factory for NIRS Analysis Backend.
"""
from flask import Flask, jsonify
from flask_cors import CORS
import os

from app.config import Config

def create_app(config_class=Config):
    """Create, configure and return the Flask application instance."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure necessary folders exist
    os.makedirs(app.config.get('UPLOAD_FOLDER', 'uploads'), exist_ok=True)
    os.makedirs(app.config.get('PROCESSED_DATA_FOLDER', 'data/processed'), exist_ok=True)
    os.makedirs(app.config.get('TEMP_DATA_FOLDER', 'data/temp'), exist_ok=True)

    # Enable CORS for frontend clients
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "*"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

    # Root index route
    @app.route('/')
    def index():
        return jsonify({
            'status': 'online',
            'service': 'NIRS Analysis Backend API',
            'version': '2.0',
            'endpoints': [
                '/api/health',
                '/api/upload',
                '/api/files',
                '/api/available_activities',
                '/api/analyze',
                '/api/temporal_validation'
            ]
        })

    # Register API blueprint
    from app.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    # Global HTTP headers (prevent caching & ensure CORS headers)
    @app.after_request
    def add_security_and_cors_headers(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    # Global Error Handlers
    @app.errorhandler(413)
    def file_too_large(error):
        return jsonify({'error': 'Uploaded file exceeds maximum allowed limit (64MB)'}), 413

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error during analysis'}), 500

    return app
