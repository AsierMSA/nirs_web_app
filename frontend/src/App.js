import React, { useState, useEffect } from 'react';
import FileUploader from './components/FileUploader';
import FileList from './components/FileList';
import ActivitySelector from './components/ActivitySelector';
import PlotViewer from './components/PlotViewer';
import './styles/App.css';
import InterpretationViewer from './components/InterpretationViewer';
import FeatureImportanceViewer from './components/FeatureImportanceViewer';
import TemporalValidationResults from './components/TemporalValidationResults';
import { fetchAvailableFiles, analyzeFile, runTemporalValidation } from './api/apiService';

function App() {
  // State management
  const [files, setFiles] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedActivities, setSelectedActivities] = useState({});
  const [plots, setPlots] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisProgress, setAnalysisProgress] = useState({});
  
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
  const handleFileDelete = async (fileId) => {
    try {
      // Call API to delete the file (if you have a delete endpoint)
      // await deleteFile(fileId);
      
      // Remove from selected files
      if (selectedFiles.includes(fileId)) {
        setSelectedFiles(selectedFiles.filter(id => id !== fileId));
      }
      
      // Remove from selected activities
      if (selectedActivities[fileId]) {
        setSelectedActivities(prev => {
          const updated = { ...prev };
          delete updated[fileId];
          return updated;
        });
      }
      
      // Remove from files list
      setFiles(prev => prev.filter(file => file.id !== fileId));
    } catch (err) {
      console.error('Error deleting file:', err);
      setError('Failed to delete file');
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
    
    // Initialize progress state for each file
    const initialProgress = {};
    selectedFiles.forEach(fileId => {
      if (selectedActivities[fileId]?.length > 0) {
        initialProgress[fileId] = { 
          status: 'preparing',
          message: 'Preparing analysis...',
          progress: 0 
        };
      }
    });
    setAnalysisProgress(initialProgress);
    
    try {
      const results = {};
      
      for (const fileId of selectedFiles) {
        if (selectedActivities[fileId] && selectedActivities[fileId].length > 0) {
          // Update progress - starting analysis
          setAnalysisProgress(prev => ({
            ...prev, 
            [fileId]: {
              status: 'loading',
              message: 'Loading data and extracting features...',
              progress: 10
            }
          }));
          
          // Wait a little to show first stage
          await new Promise(r => setTimeout(r, 500));
          
          // Update progress - SVM tuning
          setAnalysisProgress(prev => ({
            ...prev, 
            [fileId]: {
              status: 'tuning',
              message: 'Tuning SVM classifier...',
              progress: 30
            }
          }));
          
          await new Promise(r => setTimeout(r, 500));
          
          // Update progress - RandomForest tuning
          setAnalysisProgress(prev => ({
            ...prev, 
            [fileId]: {
              status: 'tuning',
              message: 'Tuning RandomForest classifier...',
              progress: 60
            }
          }));
          
          await new Promise(r => setTimeout(r, 500));
          
          // Update progress - Ridge tuning  
          setAnalysisProgress(prev => ({
            ...prev, 
            [fileId]: {
              status: 'tuning',
              message: 'Tuning Ridge classifier...',
              progress: 80
            }
          }));
          
          const fileResult = await analyzeFile(fileId, selectedActivities[fileId]);
          
          // Update progress - completed
          setAnalysisProgress(prev => ({
            ...prev, 
            [fileId]: {
              status: 'completed',
              message: 'Analysis completed',
              progress: 100
            }
          }));
          
          // Log the most important feature
          if (fileResult.features?.top_features?.length > 0) {
            // Log feature details
            const feature = fileResult.features.top_features[0];
            const region = feature.split('_')[0] || 'unknown';
            const wavelength = feature.includes('850') ? '850nm (oxyHb)' : 
                             feature.includes('760') ? '760nm (deoxyHb)' : 'unknown';
            
            console.log(`   Region: ${region}`);
            console.log(`   Wavelength: ${wavelength}`);
            console.log(`   Relative importance: Highest F-score among all features`);
          } else {
            console.log(`No important features found for ${fileId}`);
          }
          
          results[fileId] = fileResult;
        }
      }
      
      setPlots(results);
    } catch (err) {
      console.error('Analysis error:', err);
      setError('An error occurred during analysis');
      
      // Mark all analyses in progress as failed
      setAnalysisProgress(prev => {
        const updated = {...prev};
        Object.keys(updated).forEach(fileId => {
          if (updated[fileId].status !== 'completed') {
            updated[fileId] = {
              status: 'error',
              message: 'Analysis failed',
              progress: 0
            };
          }
        });
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };
  // Handle Temporal Validation request
  const handleTemporalValidation = async () => {
    setLoading(true);
    setError(null);
    setPlots({}); // Clear previous results or decide if you want to merge/keep them

    const initialProgress = {};
    selectedFiles.forEach(fileId => {
      if (selectedActivities[fileId]?.length > 0) {
        initialProgress[fileId] = {
          status: 'validating', // Specific status for validation
          message: 'Preparing temporal validation...',
          progress: 0
        };
      }
    });
    setAnalysisProgress(initialProgress);

    try {
      const validationResults = {};
      for (const fileId of selectedFiles) {
        if (selectedActivities[fileId] && selectedActivities[fileId].length > 0) {
          setAnalysisProgress(prev => ({
            ...prev,
            [fileId]: {
              status: 'validating',
              message: 'Running temporal bias checks...',
              progress: 50 // Mid-point progress
            }
          }));

          // Call the temporal validation API endpoint
          const fileResult = await runTemporalValidation(fileId, selectedActivities[fileId]);

          setAnalysisProgress(prev => ({
            ...prev,
            [fileId]: {
              status: 'completed', // Mark as completed after validation
              message: 'Temporal validation completed',
              progress: 100
            }
          }));

          validationResults[fileId] = fileResult; // Store validation results
        }
      }
      setPlots(validationResults); // Update state with validation results
    } catch (err) {
      console.error('Temporal validation error:', err);
      setError('An error occurred during temporal validation');
      // Mark ongoing validations as error
      setAnalysisProgress(prev => {
        const updated = {...prev};
        Object.keys(updated).forEach(fileId => {
          if (updated[fileId].status === 'validating') {
            updated[fileId] = {
              status: 'error',
              message: 'Validation failed',
              progress: 0
            };
          }
        });
        return updated;
      });
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
          onDeleteFile={handleFileDelete}
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
  <div className="button-group">
    <button 
      className="analyze-button" 
      onClick={handleAnalyze}
      disabled={loading}
    >
      {loading ? 'Analyzing...' : 'Analyze Selected Data'}
    </button>
    
    <button 
      className="validate-button" 
      disabled={loading} // This is correct, using 'loading' instead of 'isLoading'
      onClick={handleTemporalValidation}
    >
      Validate Against Temporal Bias
    </button>
  </div>
)}
          
          {error && <p className="error-message">{error}</p>}
          
          {/* Progress indicator for each file */}
          {Object.entries(analysisProgress).length > 0 && (
            <div className="analysis-progress">
              {Object.entries(analysisProgress).map(([fileId, progress]) => (
                <div key={fileId} className="file-progress">
                  <div className="file-progress-header">
                    <span>{files.find(file => file.id === fileId)?.name || fileId}</span>
                    <span className={`status-badge ${progress.status}`}>
                      {progress.status === 'tuning' && 
                        <span className="loading-dots">⚙️ Optimizing parameters<span>.</span><span>.</span><span>.</span></span>
                      }
                      {progress.status === 'validating' && 
                        <span className="loading-dots">🛡️ Validating<span>.</span><span>.</span><span>.</span></span>
                      }
                      {progress.status === 'loading' && '🔍 Analyzing'}
                      {progress.status === 'completed' && '✅ Completed'}
                      {progress.status === 'error' && '❌ Error'}
                    </span>
                  </div>
                  <div className="progress-bar-container">
                    <div 
                      className={`progress-bar ${progress.status}`} 
                      style={{width: `${progress.progress}%`}}
                    ></div>
                  </div>
                  <div className="progress-message">{progress.message}</div>
                </div>
              ))}
            </div>
          )}
        </section>
        
        <section className="results-section">
          <h2>Analysis Results</h2>
          {Object.entries(plots).map(([fileId, plotData]) => (
            <div key={fileId} className="result-container">
              <PlotViewer 
                key={`plot-${fileId}`}
                fileName={files.find(file => file.id === fileId)?.name || fileId}
                plotData={plotData} 
              />
            {/* Add the FeatureImportanceViewer here */}
            {plotData.plots?.feature_importance && plotData.features?.top_features && (
              <FeatureImportanceViewer
                key={`feat-imp-${fileId}`}
                featureImportanceData={plotData.plots.feature_importance}
                topFeatures={plotData.features.top_features}
              />
            )}
            {plotData.temporal_validation && (
            <TemporalValidationResults 
              validationData={plotData.temporal_validation} 
            />
          )}
            <InterpretationViewer 
              key={`interp-${fileId}`}
              interpretationData={plotData.interpretation || {}}
              topFeatures={plotData.features?.top_features || []}
            />
            </div>
          ))}

        </section>
      </main>
    </div>
  );
}

export default App;