"""
Unit tests for API routes and endpoint validation.
"""
import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_root_endpoint(client):
    """Test index endpoint returns status online."""
    response = client.get('/')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['status'] == 'online'
    assert 'endpoints' in json_data

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['status'] == 'healthy'

def test_get_files(client):
    """Test listing uploaded files."""
    response = client.get('/api/files')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'files' in json_data
    assert isinstance(json_data['files'], list)

def test_upload_no_file(client):
    """Test upload endpoint with empty request."""
    response = client.post('/api/upload')
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data

def test_upload_invalid_extension(client):
    """Test upload endpoint with invalid file type."""
    import io
    data = {
        'file': (io.BytesIO(b"dummy text content"), 'invalid_file.txt')
    }
    response = client.post('/api/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert 'Invalid file type' in response.get_json()['error']

def test_available_activities_missing_param(client):
    """Test available activities without file_id query param."""
    response = client.get('/api/available_activities')
    assert response.status_code == 400

def test_analyze_empty_body(client):
    """Test analyze endpoint with empty payload."""
    response = client.post('/api/analyze', json={})
    assert response.status_code == 400
