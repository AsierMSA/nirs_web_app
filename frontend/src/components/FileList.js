import React, { useState } from 'react';
import '../styles/components.css';

function FileList({ files, selectedFiles, onSelectFile, onDeleteFile }) {
  const [confirmDelete, setConfirmDelete] = useState(null);
  
  if (!files || files.length === 0) {
    return <p className="info-text">No files available. Upload NIRS files to start.</p>;
  }
  
  const handleDelete = (fileId) => {
    // Skip confirmation and delete directly, or show confirmation dialog
    if (onDeleteFile) {
      onDeleteFile(fileId);
      // If the deleted file was selected, it will be automatically removed from selection
      // in App.js handleFileDelete function
    }
  };

  return (
    <div className="file-list">
      <ul>
        {files.map(file => (
          <li key={file.id} className="file-item">
            <label className="file-label">
              <input
                type="checkbox"
                checked={selectedFiles.includes(file.id)}
                onChange={() => onSelectFile(file.id)}
              />
              <span className="file-name">{file.name}</span>
            </label>
            <button
              className="delete-button"
              onClick={() => handleDelete(file.id)}
              title="Delete file"
            >
              <span>🗑️</span>
            </button>
          </li>
        ))}
      </ul>

      {confirmDelete && (
        <div className="confirmation-dialog">
          <p>Are you sure you want to delete this file?</p>
          <div className="confirmation-buttons">
            <button onClick={() => {
              onDeleteFile(confirmDelete);
              setConfirmDelete(null);
            }}>
              Yes
            </button>
            <button onClick={() => setConfirmDelete(null)}>No</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default FileList;