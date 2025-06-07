import React from 'react';
import '../styles/components.css';

function FileList({ files, selectedFiles, onSelectFile, onDeleteFile }) {
  if (!files || files.length === 0) {
    return <p className="info-text">No files available</p>;
  }

  return (
    <div className="file-list">
      {files.map((file, index) => {
        const fileId = file.id || `file_${index}`;
        const fileName = file.name || `File ${index + 1}`;
        
        return (
          <div key={fileId} className="file-item">
            <input
              type="checkbox"
              id={`file-${fileId}`}
              checked={selectedFiles.includes(fileId)}
              onChange={() => onSelectFile(fileId)}
            />
            <label htmlFor={`file-${fileId}`} className="file-name">
              {fileName}
              {/* Debug info - remove in production */}
              {process.env.NODE_ENV === 'development' && file.originalData && (
                <small style={{ display: 'block', color: '#666', fontSize: '0.8em' }}>
                  Debug ID: {fileId} | Original: {JSON.stringify(file.originalData)}
                </small>
              )}
            </label>
            <button 
              className="delete-button"
              onClick={() => onDeleteFile(fileId)}
              title="Delete file"
            >
              🗑️
            </button>
          </div>
        );
      })}
    </div>
  );
}

export default FileList;