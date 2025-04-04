# This file initializes the api package, preparing it for route and validator definitions.

from flask import Blueprint

# Create a blueprint for the API
api_bp = Blueprint('api', __name__)

# Import routes and validators to register them with the blueprint
from . import routes, validators