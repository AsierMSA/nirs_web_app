// Base API URL - update this to match your backend URL
const API_BASE_URL = 'http://localhost:5000';

/**
 * Upload a NIRS file to the backend
 */
export async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to upload file');
  }
  
  const data = await response.json();
  return {
    id: data.file_id,
    name: data.filename
  };
}

/**
 * Fetch available files from the backend
 */
export async function fetchAvailableFiles() {
  const response = await fetch(`${API_BASE_URL}/api/files`);
  
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to fetch available files');
  }
  
  const data = await response.json();
  return data.files.map(file => ({
    id: file.file_id,
    name: file.filename
  }));
}

/**
 * Fetch activities available in a specific file
 */
export async function fetchFileActivities(fileId) {
  const response = await fetch(`${API_BASE_URL}/api/available_activities?file_id=${fileId}`);
  
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Failed to fetch file activities');
  }
  
  const data = await response.json();
  return data.activities;
}

/**
 * Run analysis on a specific file with selected activities
 */
export async function analyzeFile(fileId, activities) {
  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      file_id: fileId,
      activities: activities
    })
  });
  
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || 'Analysis failed');
  }
  
  return response.json();
}