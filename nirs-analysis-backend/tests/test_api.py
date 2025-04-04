import pytest
from app.api.routes import upload_nirs_file, get_analysis_results
from fastapi.testclient import TestClient

# Initialize the FastAPI test client
client = TestClient(app)

def test_upload_nirs_file():
    # Test uploading a valid NIRS file
    response = client.post("/upload", files={"file": ("test_file.fif", open("data/uploads/test_file.fif", "rb"))})
    assert response.status_code == 200
    assert "Analysis started" in response.json().get("message")

    # Test uploading an invalid file type
    response = client.post("/upload", files={"file": ("test_file.txt", open("data/uploads/test_file.txt", "rb"))})
    assert response.status_code == 400
    assert "Invalid file type" in response.json().get("detail")

def test_get_analysis_results():
    # Assuming an analysis has been performed and results are available
    response = client.get("/results/1")  # Replace '1' with a valid analysis ID
    assert response.status_code == 200
    assert "accuracy" in response.json()
    assert "feature_importance" in response.json()

    # Test getting results for a non-existent analysis ID
    response = client.get("/results/999")  # Assuming 999 does not exist
    assert response.status_code == 404
    assert "Analysis not found" in response.json().get("detail")