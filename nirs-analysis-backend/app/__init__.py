# This file initializes the app package and can be used to set up application-level configurations.

from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Load configuration settings
    app.config.from_object('app.config.Config')
    
    # Register blueprints for API routes
    from app.api.routes import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')
    
    return app