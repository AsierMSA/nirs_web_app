
import React, { useState, useEffect } from 'react';
import FileUploader from './components/FileUploader';
import FileList from './components/FileList';
import ActivitySelector from './components/ActivitySelector';
import PlotViewer from './components/PlotViewer';
import InterpretationViewer from './components/InterpretationViewer';
import TemporalValidationResults from './components/TemporalValidationResults';
import { fetchAvailableFiles, analyzeFile, runTemporalValidation } from './api/apiService';
import './styles/App.css';

function App() {
  const [files, setFiles] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedActivities, setSelectedActivities] = useState({});
  const [analysisResults, setAnalysisResults] = useState({});
  const [temporalValidationResults, setTemporalValidationResults] = useState({});
  const [loading, setLoading] = useState({
    files: false,
    analysis: {},
    temporalValidation: {}
  });
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    setLoading(prev => ({ ...prev, files: true }));
    setError(null);
    
    try {
      console.log('📁 Fetching available files...');
      const processedFiles = await fetchAvailableFiles();
      
      console.log('✅ Files received:', processedFiles);
      setFiles(processedFiles);
      
    } catch (err) {
      console.error('❌ Error fetching files:', err);
      setError(`Failed to fetch files: ${err.message}`);
    } finally {
      setLoading(prev => ({ ...prev, files: false }));
    }
  };

  const handleFileUpload = (newFile) => {
    console.log('📤 New file uploaded:', newFile);
    
    const processedFile = {
      id: newFile.id || `uploaded_${Date.now()}`,
      name: newFile.name || 'Unknown File',
      ...newFile
    };
    
    setFiles(prev => [...prev, processedFile]);
    setError(null);
  };

  const handleSelectFile = (fileId) => {
    console.log('🔍 Selecting file with ID:', fileId);
    
    if (!fileId || fileId === 'undefined') {
      console.error('❌ Invalid file ID:', fileId);
      setError('Invalid file selection');
      return;
    }
    
    setSelectedFiles(prev => 
      prev.includes(fileId) 
        ? prev.filter(id => id !== fileId)
        : [...prev, fileId]
    );
  };

  const handleDeleteFile = (fileId) => {
    setFiles(prev => prev.filter(file => file.id !== fileId));
    setSelectedFiles(prev => prev.filter(id => id !== fileId));
    setSelectedActivities(prev => {
      const { [fileId]: deleted, ...rest } = prev;
      return rest;
    });
    setAnalysisResults(prev => {
      const { [fileId]: deleted, ...rest } = prev;
      return rest;
    });
    setTemporalValidationResults(prev => {
      const { [fileId]: deleted, ...rest } = prev;
      return rest;
    });
  };

  const handleSelectActivities = (fileId, activities) => {
    console.log('📋 Activities selected for', fileId, ':', activities);
    setSelectedActivities(prev => ({ ...prev, [fileId]: activities }));
  };

  const handleAnalyze = async (fileId) => {
    if (!fileId || fileId === 'undefined') {
      setError('Invalid file ID for analysis');
      return;
    }
    
    if (!selectedActivities[fileId] || selectedActivities[fileId].length === 0) {
      setError(`Please select activities for the file before analyzing.`);
      return;
    }
    
    setLoading(prev => ({ ...prev, analysis: { ...prev.analysis, [fileId]: true } }));
    setError(null);
    
    try {
      console.log(`🔄 Starting analysis for ${fileId} with activities:`, selectedActivities[fileId]);
      
      const results = await analyzeFile(fileId, selectedActivities[fileId]);
      
      console.log(`📊 Analysis results for ${fileId}:`, {
        hasPlots: !!results.plots,
        plotKeys: results.plots ? Object.keys(results.plots) : [],
        hasMLResults: !!results.ml_results,
        mlResultsKeys: results.ml_results ? Object.keys(results.ml_results) : [],
        hasFeatures: !!results.features,
        featureCount: results.features?.feature_count || 0
      });
      
      setAnalysisResults(prev => ({ ...prev, [fileId]: results }));
      setError(null);
      
    } catch (err) {
      console.error(`❌ Analysis failed for ${fileId}:`, err);
      setError(`Analysis failed: ${err.message}`);
      setAnalysisResults(prev => ({ 
        ...prev, 
        [fileId]: { error: `Analysis failed: ${err.message}` } 
      }));
    } finally {
      setLoading(prev => ({ ...prev, analysis: { ...prev.analysis, [fileId]: false } }));
    }
  };

  const handleTemporalValidation = async (fileId) => {
    if (!fileId || fileId === 'undefined') {
      setError('Invalid file ID for validation');
      return;
    }
    
    if (!selectedActivities[fileId] || selectedActivities[fileId].length === 0) {
      setError(`Please select activities for the file before validation.`);
      return;
    }
    
    setLoading(prev => ({ ...prev, temporalValidation: { ...prev.temporalValidation, [fileId]: true } }));
    setError(null);
    
    try {
      const results = await runTemporalValidation(fileId, selectedActivities[fileId]);
      setTemporalValidationResults(prev => ({ ...prev, [fileId]: results }));
    } catch (err) {
      setError(`Temporal validation failed: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(prev => ({ ...prev, temporalValidation: { ...prev.temporalValidation, [fileId]: false } }));
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>NIRS Data Analysis Platform</h1>
        <p>Advanced neuroimaging signal processing and machine learning analysis</p>
      </header>

      <main className="app-content">
        <section>
          <h2>📁 Upload Files</h2>
          <FileUploader onFileUpload={handleFileUpload} />
        </section>

        <section>
          <h2>📂 Available Files</h2>
          {loading.files ? (
            <p className="info-text">Loading files...</p>
          ) : (
            <FileList 
              files={files}
              selectedFiles={selectedFiles}
              onSelectFile={handleSelectFile}
              onDeleteFile={handleDeleteFile}
            />
          )}
        </section>

        {selectedFiles.length > 0 && (
          <section style={{ gridColumn: '1 / -1' }}>
            <h2>🎯 Select Activities</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '20px' }}>
              {selectedFiles.map(fileId => {
                const file = files.find(f => f.id === fileId);
                
                if (!file) {
                  console.warn(`⚠️ File not found for ID: ${fileId}`);
                  return (
                    <div key={fileId} className="error-message">
                      ❌ File with ID {fileId} not found
                    </div>
                  );
                }
                
                return (
                  <ActivitySelector
                    key={fileId}
                    fileId={fileId}
                    fileName={file.name}
                    onSelectActivities={(activities) => handleSelectActivities(fileId, activities)}
                  />
                );
              })}
            </div>
          </section>
        )}

        {selectedFiles.length > 0 && Object.keys(selectedActivities).length > 0 && (
          <section style={{ gridColumn: '1 / -1' }}>
            <h2>🔬 Analysis</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '20px' }}>
              {selectedFiles.map(fileId => {
                const file = files.find(f => f.id === fileId);
                const hasActivities = selectedActivities[fileId] && selectedActivities[fileId].length > 0;
                
                if (!file || !hasActivities) return null;
                
                return (
                  <div key={fileId} className="analysis-item">
                    <h3>📄 {file.name}</h3>
                    <p><strong>Selected activities:</strong> {selectedActivities[fileId].join(', ')}</p>
                    
                    <div className="button-group">
                      <button 
                        className="analyze-button"
                        onClick={() => handleAnalyze(fileId)}
                        disabled={loading.analysis[fileId]}
                      >
                        {loading.analysis[fileId] ? (
                          <>
                            <span className="loading-dots">
                              <span>.</span><span>.</span><span>.</span>
                            </span>
                            Analyzing
                          </>
                        ) : (
                          '📊 Analyze'
                        )}
                      </button>
                      
                      <button 
                        className="analyze-button"
                        onClick={() => handleTemporalValidation(fileId)}
                        disabled={loading.temporalValidation[fileId]}
                        style={{ backgroundColor: '#e67e22' }}
                      >
                        {loading.temporalValidation[fileId] ? (
                          <>
                            <span className="loading-dots">
                              <span>.</span><span>.</span><span>.</span>
                            </span>
                            Validating
                          </>
                        ) : (
                          '⏱️ Temporal Validation'
                        )}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}
      </main>

      {error && (
        <div className="error-message">
          <strong>⚠️ Error:</strong> {error}
        </div>
      )}

      <section className="results-section">
        <h2>📈 Results</h2>
        {Object.entries(analysisResults).length === 0 ? (
          <p className="info-text">No analysis results yet. Run an analysis to see results here.</p>
        ) : (
          Object.entries(analysisResults).map(([fileId, results]) => {
            const file = files.find(f => f.id === fileId);
            if (!file) return null;

            return (
              <div key={fileId} style={{ marginBottom: '40px' }}>
                <PlotViewer 
                  fileName={file.name} 
                  plotData={results}
                />
                
                {results.interpretation && (
                  <InterpretationViewer 
                    interpretationData={results.interpretation}
                    topFeatures={results.features?.top_features || []}
                  />
                )}
                
                {temporalValidationResults[fileId] && (
                  <TemporalValidationResults 
                    validationData={temporalValidationResults[fileId]}
                  />
                )}
              </div>
            );
          })
        )}
      </section>
    </div>
  );
}

export default App;