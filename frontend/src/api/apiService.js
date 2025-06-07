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
  console.log('📤 Upload response:', data);
  
  return {
    id: data.file_id || data.id || data.filename || data.name,
    name: data.filename || data.name || data.file_id || data.id
  };
}

/**
 * Fetch available files from the backend
 */
export async function fetchAvailableFiles() {
  try {
    console.log('🌐 Making request to:', `${API_BASE_URL}/api/files`);
    const response = await fetch(`${API_BASE_URL}/api/files`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('🔍 Raw API response:', data);
    
    // ✅ MANEJAR DIFERENTES ESTRUCTURAS DE RESPUESTA
    let files = [];
    
    if (data.files && Array.isArray(data.files)) {
      files = data.files;
    } else if (Array.isArray(data)) {
      files = data;
    } else {
      console.error('❌ Unexpected response format:', data);
      return [];
    }
    
    console.log('🔍 Raw files array:', files);
    
    // ✅ PROCESAR ARCHIVOS CON ESTRUCTURA FLEXIBLE
    const processedFiles = files.map((file, index) => {
      console.log(`📁 Processing file ${index}:`, file);
      
      // Intentar todas las posibles propiedades para ID y nombre
      const possibleId = file.file_id || file.id || file.filename || file.name || file.path;
      const possibleName = file.filename || file.name || file.file_name || file.fileName || file.file_id || file.id;
      
      if (!possibleId && !possibleName) {
        console.warn(`⚠️ File ${index} has no identifiable properties:`, file);
        return {
          id: `file_${index}`,
          name: `File ${index + 1}`,
          originalData: file
        };
      }
      
      return {
        id: possibleId || `file_${index}`,
        name: possibleName || possibleId || `File ${index + 1}`,
        originalData: file
      };
    });
    
    console.log('✅ Processed files:', processedFiles);
    return processedFiles;
    
  } catch (error) {
    console.error('❌ Error in fetchAvailableFiles:', error);
    throw error;
  }
}

/**
 * Fetch activities available in a specific file
 */
export async function fetchFileActivities(fileId) {
  try {
    console.log('🔍 Fetching activities for fileId:', fileId);
    
    if (!fileId || fileId === 'undefined') {
      throw new Error('Invalid file ID provided');
    }
    
    const response = await fetch(`${API_BASE_URL}/api/available_activities?file_id=${encodeURIComponent(fileId)}`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || 'File not found');
    }
    
    const data = await response.json();
    return data.activities || [];
  } catch (error) {
    console.error('❌ Error fetching activities:', error);
    throw error;
  }
}

/**
 * Run analysis on a specific file with selected activities
 */
export const analyzeFile = async (fileId, activities) => {
  try {
    console.log('🔬 Starting analysis for:', { fileId, activities });
    
    if (!fileId || fileId === 'undefined') {
      throw new Error('Invalid file ID provided');
    }
    
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        file_id: fileId, 
        activities 
      }),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[ERROR] Server response (${response.status}):`, errorText);
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`[ERROR] API call failed for file ${fileId}:`, error);
    throw error;
  }
};

/**
 * Run temporal validation analysis on a specific file with selected activities
 */
export const runTemporalValidation = async (fileId, activities) => {
  try {
    console.log('⏱️ Starting temporal validation for:', { fileId, activities });
    
    if (!fileId || fileId === 'undefined') {
      throw new Error('Invalid file ID provided');
    }
    
    const response = await fetch(`${API_BASE_URL}/api/temporal_validation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        file_id: fileId,
        activities: activities
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`[ERROR] Temporal validation failed for file ${fileId}:`, error);
    throw error;
  }
};