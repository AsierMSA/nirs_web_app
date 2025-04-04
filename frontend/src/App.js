import React, { useState, useEffect } from 'react';
import FileUploader from './components/FileUploader';
import FileList from './components/FileList';
import ActivitySelector from './components/ActivitySelector';
import PlotViewer from './components/PlotViewer';
import { fetchAvailableFiles, analyzeFile } from './api/apiService';
import './styles/App.css';

function App() {
  // State management
  const [files, setFiles] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [activities, setActivities] = useState({});
  const [selectedActivities, setSelectedActivities] = useState({});
  const [plots, setPlots] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load available files on component mount
  useEffect(() => {
    const loadFiles = async () => {
      try {
        const availableFiles = await fetchAvailableFiles();
        setFiles(availableFiles);
      } catch (err) {
        console.error('Error loading files:', err);
        setError('Failed to load available files');
      }
    };
    
    loadFiles();
  }, []);

  // Handle file selection
  const handleFileSelect = (fileId) => {
    const isSelected = selectedFiles.includes(fileId);
    
    if (isSelected) {
      setSelectedFiles(selectedFiles.filter(id => id !== fileId));
      setSelectedActivities(prev => {
        const updated = { ...prev };
        delete updated[fileId];
        return updated;
      });
    } else {
      setSelectedFiles([...selectedFiles, fileId]);
    }
  };

  // Handle successful file upload
  const handleFileUpload = (uploadedFile) => {
    setFiles(prev => [...prev, uploadedFile]);
  };

  // Handle activity selection
  const handleActivitySelect = (fileId, selectedActivitiesList) => {
    setSelectedActivities(prev => ({
      ...prev,
      [fileId]: selectedActivitiesList
    }));
  };

  // Handle analysis request
  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setPlots({});
    
    try {
      const results = {};
      
      // Process each selected file with its selected activities
      for (const fileId of selectedFiles) {
        if (selectedActivities[fileId] && selectedActivities[fileId].length > 0) {
          const fileResult = await analyzeFile(fileId, selectedActivities[fileId]);
          results[fileId] = fileResult;
        }
      }
      
      setPlots(results);
    } catch (err) {
      console.error('Analysis error:', err);
      setError('An error occurred during analysis');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>NIRS Analysis Dashboard</h1>
      </header>
      
      <main className="app-content">
        <section className="file-section">
          <h2>NIRS Files</h2>
          <FileUploader onFileUpload={handleFileUpload} />
          <FileList 
            files={files} 
            selectedFiles={selectedFiles} 
            onSelectFile={handleFileSelect}
          />
        </section>
        
        <section className="activity-section">
          <h2>Activities</h2>
          {selectedFiles.length > 0 ? (
            selectedFiles.map(fileId => (
              <ActivitySelector
                key={fileId}
                fileId={fileId}
                fileName={files.find(file => file.id === fileId)?.name || fileId}
                onSelectActivities={(activities) => handleActivitySelect(fileId, activities)}
              />
            ))
          ) : (
            <p className="info-text">Select files to view available activities</p>
          )}
          
          {selectedFiles.length > 0 && Object.keys(selectedActivities).length > 0 && (
            <button 
              className="analyze-button" 
              onClick={handleAnalyze}
              disabled={loading}
            >
              {loading ? 'Analyzing...' : 'Analyze Selected Data'}
            </button>
          )}
          
          {error && <p className="error-message">{error}</p>}
        </section>
        
        <section className="results-section">
          <h2>Analysis Results</h2>
          {Object.keys(plots).length > 0 ? (
            Object.entries(plots).map(([fileId, plotData]) => (
              <PlotViewer 
                key={fileId}
                fileName={files.find(file => file.id === fileId)?.name || fileId}
                plotData={plotData} 
              />
            ))
          ) : (
            <p className="info-text">Run analysis to view results</p>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;