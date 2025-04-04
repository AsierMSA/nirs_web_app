# NIRS Analysis Backend

This project is a backend application designed for analyzing Near-Infrared Spectroscopy (NIRS) data. It provides an API for uploading NIRS files, performing analysis, and retrieving results, including visualizations.

## Project Structure

The project is organized into several directories and files:

- **app/**: Contains the main application code.
  - **api/**: Defines the API routes and validation logic.
  - **config.py**: Holds configuration settings for the application.
  - **core/**: Contains the core logic for data analysis and plotting.
  - **models/**: Defines data models for storing analysis results.
  - **utils/**: Includes utility functions for file handling and response formatting.

- **data/**: Directory for storing data files.
  - **processed/**: Stores processed NIRS data files.
  - **temp/**: Holds temporary files during processing.
  - **uploads/**: Used for storing uploaded NIRS files before analysis.

- **tests/**: Contains unit tests for the application.
  - **test_api.py**: Tests for API routes.
  - **test_analyzer.py**: Tests for analysis functions.

- **.env.example**: Template for environment variables needed for the application.

- **.gitignore**: Specifies files and directories to be ignored by version control.

- **requirements.txt**: Lists dependencies required for the project.

- **run.py**: Entry point for running the application.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd nirs-analysis-backend
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Copy the `.env.example` file to `.env` and update the variables as needed.

5. **Run the Application**:
   ```bash
   python run.py
   ```

## Usage

- Use the API endpoints defined in `app/api/routes.py` to upload NIRS files and retrieve analysis results.
- The application will process the uploaded files and generate plots that can be sent to the frontend for visualization.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.