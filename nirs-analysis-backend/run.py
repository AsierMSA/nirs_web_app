# Contents of /nirs-analysis-backend/nirs-analysis-backend/run.py

from app import create_app

# Initialize the application
app = create_app()

if __name__ == "__main__":
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)  # Set debug to False in production