import React from 'react';
import '../styles/components.css';

function FileList({ files, selectedFiles, onSelectFile }) {
  if (!files || files.length === 0) {
    return <p className="info-text">No files available. Upload NIRS files to start.</p>;
  }

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
          </li>
        ))}
      </ul>
    </div>
  );
}

export default FileList;