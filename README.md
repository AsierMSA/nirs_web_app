
*(Note: The `nirs-toolbox` directory containing MATLAB code is also present but seems separate from the core web application described here.)*

## Setup and Installation

### Prerequisites

*   Python (3.11 or higher recommended)
*   `pip` (Python package installer)
*   Node.js and `npm` (or `yarn`)

### Backend Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd nirs_web_app/nirs-analysis-backend
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Optional) Environment Variables:** If required, create a `.env` file based on `.env.example` and configure necessary variables.

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd ../frontend
    ```
    *(Assuming you are in the `nirs-analysis-backend` directory)*
    *or*
    ```bash
    cd nirs_web_app/frontend
    ```
    *(Assuming you are in the main project root)*

2.  **Install Node dependencies:**
    ```bash
    npm install
    # or if using yarn
    # yarn install
    ```

## Running the Application

1.  **Start the Backend Server:**
    *   Make sure your virtual environment is activated.
    *   Navigate to the `nirs-analysis-backend` directory.
    *   Run the Flask application:
        ```bash
        flask run
        # or potentially: python run.py
        ```
    *   By default, the backend usually runs on `http://127.0.0.1:5000`.

2.  **Start the Frontend Development Server:**
    *   Navigate to the `frontend` directory.
    *   Run the React development server:
        ```bash
        npm start
        # or if using yarn
        # yarn start
        ```
    *   This will usually open the application automatically in your web browser at `http://localhost:3000`.

## Usage

1.  Open your web browser and navigate to the frontend URL (typically `http://localhost:3000`).
2.  Use the interface to upload your NIRS data file.
3.  The backend will process the data, perform feature extraction, run machine learning classification, and generate results.
4.  View the analysis results, visualizations (feature importance, classifier comparison, confusion matrix, learning curve), and interpretations displayed on the frontend.

## Contributing

Contributions are welcome! Please follow standard Git workflow practices (fork, branch, pull request). Ensure code is well-documented and tests are added/updated where applicable.

## License

This project is licensed under the MIT License. See the `LICENSE` file (if available) for more details.
