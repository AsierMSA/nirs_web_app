import React, { useState } from 'react';
import { uploadFile } from '../api/apiService';
import '../styles/components.css';

function FileUploader({ onFileUpload }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    // Check if file has a valid extension
    const validExtensions = ['.fif', '.gz'];
    const hasValidExtension = validExtensions.some(ext => 
      file.name.toLowerCase().endsWith(ext) || file.name.toLowerCase().endsWith('fif.gz')
    );

    if (!hasValidExtension) {
      setError('Invalid file type. Only .fif or .fif.gz files are allowed.');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const uploadedFile = await uploadFile(file);
      setFile(null);
      onFileUpload(uploadedFile);
      // Reset file input
      document.getElementById('file-input').value = '';
    } catch (err) {
      console.error('Upload error:', err);
      setError('Failed to upload file');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="file-uploader">
      <div className="file-input-container">
        <input
          id="file-input"
          type="file"
          onChange={handleFileChange}
          accept=".fif,.gz"
          disabled={uploading}
        />
        <button 
          className="upload-button"
          onClick={handleUpload} 
          disabled={!file || uploading}
        >
          {uploading ? 'Uploading...' : 'Upload'}
        </button>
      </div>
      
      {file && (
        <div className="file-info">
          <p>Selected: {file.name} ({Math.round(file.size / 1024)} KB)</p>
        </div>
      )}
      
      {error && <p className="error-message">{error}</p>}
    </div>
  );
}

export default FileUploader;