import pytest
from app.core.analyzer import analyze_nirs_data

def test_analyze_nirs_data_valid_file():
    # Test with a valid NIRS file and expected activities
    raw_file_path = "data/uploads/test_valid_file.fif.gz"
    activity_names = ["Finger Sequencing", "Simple Tapping"]
    
    results = analyze_nirs_data(raw_file_path, activity_names)
    
    # Check if results contain expected keys
    assert 'accuracy' in results
    assert 'best_classifier' in results
    assert 'features' in results
    assert 'region_importance' in results

def test_analyze_nirs_data_invalid_file():
    # Test with an invalid NIRS file path
    raw_file_path = "data/uploads/non_existent_file.fif.gz"
    activity_names = ["Finger Sequencing", "Simple Tapping"]
    
    results = analyze_nirs_data(raw_file_path, activity_names)
    
    # Check if results indicate failure
    assert results['accuracy'] is None
    assert results['best_classifier'] is None
    assert results['features'] is None
    assert results['region_importance'] == {}

def test_analyze_nirs_data_no_activities():
    # Test with a valid NIRS file but no matching activities
    raw_file_path = "data/uploads/test_valid_file.fif.gz"
    activity_names = ["Nonexistent Activity"]
    
    results = analyze_nirs_data(raw_file_path, activity_names)
    
    # Check if results indicate no activities found
    assert results['accuracy'] is None
    assert results['best_classifier'] is None
    assert results['features'] is None
    assert results['region_importance'] == {}

def test_analyze_nirs_data_partial_data():
    # Test with a valid NIRS file and some activities
    raw_file_path = "data/uploads/test_partial_data_file.fif.gz"
    activity_names = ["Finger Sequencing", "Simple Tapping", "Nonexistent Activity"]
    
    results = analyze_nirs_data(raw_file_path, activity_names)
    
    # Check if results contain expected keys and accuracy is calculated
    assert 'accuracy' in results
    assert results['accuracy'] is not None
    assert 'best_classifier' in results
    assert 'features' in results
    assert 'region_importance' in results

# Additional tests can be added here for more comprehensive coverage.