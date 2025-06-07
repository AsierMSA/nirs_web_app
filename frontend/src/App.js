import React, { useState, useEffect } from 'react';
import FileUploader from './components/FileUploader';
import FileList from './components/FileList';
import PlotViewer from './components/PlotViewer'; // Importa PlotViewer
import TemporalValidationResults from './components/TemporalValidationResults'; // Importa el nuevo componente
import { fetchAvailableFiles, fetchFileActivities, analyzeFile, runTemporalValidation } from './api/apiService';
import './styles/App.css'; // Estilos generales de la aplicación
import './styles/components.css'; // Estilos para componentes como FileUploader, PlotViewer, etc.

function App() {
  const [availableFiles, setAvailableFiles] = useState([]);
  const [selectedFileIds, setSelectedFileIds] = useState([]); // Puede ser un array si permites múltiples
  const [fileActivities, setFileActivities] = useState({}); // { fileId: ['act1', 'act2'], ... }
  const [selectedActivities, setSelectedActivities] = useState({}); // { fileId: ['selectedAct1'], ... }
  const [analysisResults, setAnalysisResults] = useState({}); // { fileId: { plots: {...}, features: {...}, ... }, ... }
  const [temporalValidationData, setTemporalValidationData] = useState({}); // { fileId: { validation_score: ..., p_value: ... } }
  const [loading, setLoading] = useState({ files: false, activities: {}, analysis: {}, validation: {} });
  const [error, setError] = useState(null);

  // Cargar archivos disponibles al montar el componente
  useEffect(() => {
    loadAvailableFiles();
  }, []);

  const loadAvailableFiles = async () => {
    setLoading(prev => ({ ...prev, files: true }));
    try {
      const files = await fetchAvailableFiles();
      setAvailableFiles(files);
      setError(null);
    } catch (err) {
      setError(`Error fetching files: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(prev => ({ ...prev, files: false }));
    }
  };

  const handleFileUpload = (uploadedFile) => {
    setAvailableFiles(prevFiles => {
      // Evitar duplicados si el archivo ya existe por alguna razón
      if (!prevFiles.find(f => f.id === uploadedFile.id)) {
        return [...prevFiles, uploadedFile];
      }
      return prevFiles;
    });
    // Opcionalmente, seleccionar el archivo recién subido y cargar sus actividades
    // handleFileSelect(uploadedFile.id); 
  };

  const handleFileSelect = async (fileId) => {
    const newSelectedFileIds = selectedFileIds.includes(fileId)
      ? selectedFileIds.filter(id => id !== fileId)
      : [...selectedFileIds, fileId];
    setSelectedFileIds(newSelectedFileIds);

    if (newSelectedFileIds.includes(fileId) && !fileActivities[fileId]) {
      setLoading(prev => ({ ...prev, activities: { ...prev.activities, [fileId]: true } }));
      try {
        const activities = await fetchFileActivities(fileId);
        setFileActivities(prev => ({ ...prev, [fileId]: activities }));
        // Por defecto, seleccionar todas las actividades o la primera
        if (activities.length > 0) {
          setSelectedActivities(prev => ({ ...prev, [fileId]: activities })); // Seleccionar todas por defecto
        }
        setError(null);
      } catch (err) {
        setError(`Error fetching activities for ${fileId}: ${err.message}`);
        console.error(err);
      } finally {
        setLoading(prev => ({ ...prev, activities: { ...prev.activities, [fileId]: false } }));
      }
    }
  };

  const handleActivityChange = (fileId, activity, isChecked) => {
    setSelectedActivities(prev => {
      const currentActivities = prev[fileId] || [];
      const newActivities = isChecked
        ? [...currentActivities, activity]
        : currentActivities.filter(a => a !== activity);
      return { ...prev, [fileId]: newActivities };
    });
  };

  const handleAnalyze = async (fileId) => {
    if (!selectedActivities[fileId] || selectedActivities[fileId].length === 0) {
      setError(`Please select activities for ${fileId} before analyzing.`);
      return;
    }
    setLoading(prev => ({ ...prev, analysis: { ...prev.analysis, [fileId]: true } }));
    setError(null);
    try {
      const results = await analyzeFile(fileId, selectedActivities[fileId]);
      setAnalysisResults(prev => ({ ...prev, [fileId]: results }));
    } catch (err) {
      setError(`Analysis failed for ${fileId}: ${err.message}`);
      console.error(err);
      setAnalysisResults(prev => ({ ...prev, [fileId]: { error: `Analysis failed: ${err.message}` } }));
    } finally {
      setLoading(prev => ({ ...prev, analysis: { ...prev.analysis, [fileId]: false } }));
    }
  };

  const handleTemporalValidation = async (fileId) => {
    if (!selectedActivities[fileId] || selectedActivities[fileId].length === 0) {
      setError(`Please select activities for ${fileId} before validation.`);
      return;
    }
    setLoading(prev => ({ ...prev, validation: { ...prev.validation, [fileId]: true } }));
    setError(null);
    try {
      const validationResult = await runTemporalValidation(fileId, selectedActivities[fileId]);
      // La API devuelve { temporal_validation: {...} }
      setTemporalValidationData(prev => ({ ...prev, [fileId]: validationResult.temporal_validation }));
    } catch (err) {
      setError(`Temporal validation failed for ${fileId}: ${err.message}`);
      console.error(err);
      setTemporalValidationData(prev => ({ ...prev, [fileId]: { error: `Validation failed: ${err.message}` } }));
    } finally {
      setLoading(prev => ({ ...prev, validation: { ...prev.validation, [fileId]: false } }));
    }
  };
  
  // Lógica para eliminar archivos (a implementar si es necesario)
  // const handleFileDelete = async (fileId) => { ... }


  return (
    <div className="app-container">
      <header className="app-header">
        <h1>NIRS Analysis Dashboard</h1>
      </header>

      {error && <p className="error-message">{error}</p>}

      <section className="upload-section">
        <h2>Upload NIRS File</h2>
        <FileUploader onFileUpload={handleFileUpload} />
      </section>

      <div className="app-content">
        <section className="files-section">
          <h2>Available Files</h2>
          {loading.files ? <p>Loading files...</p> : (
            <FileList
              files={availableFiles}
              selectedFiles={selectedFileIds}
              onSelectFile={handleFileSelect}
              // onDeleteFile={handleFileDelete} // Descomentar si se implementa la eliminación
            />
          )}
        </section>

        <section className="activities-section">
          <h2>Select Activities & Analyze</h2>
          {selectedFileIds.length === 0 && <p className="info-text">Select a file to see available activities.</p>}
          {selectedFileIds.map(fileId => {
            const file = availableFiles.find(f => f.id === fileId);
            return (
              <div key={fileId} className="file-analysis-block">
                <h3>{file?.name || fileId}</h3>
                {loading.activities[fileId] && <p>Loading activities...</p>}
                {fileActivities[fileId] && fileActivities[fileId].length > 0 && (
                  <div className="activity-list">
                    <h4>Activities:</h4>
                    {fileActivities[fileId].map(activity => (
                      <label key={activity} className="activity-item">
                        <input
                          type="checkbox"
                          checked={selectedActivities[fileId]?.includes(activity) || false}
                          onChange={(e) => handleActivityChange(fileId, activity, e.target.checked)}
                        />
                        {activity}
                      </label>
                    ))}
                  </div>
                )}
                {fileActivities[fileId] && fileActivities[fileId].length === 0 && (
                  <p>No activities found in this file.</p>
                )}
                <div className="button-group">
                  <button
                    className="analyze-button"
                    onClick={() => handleAnalyze(fileId)}
                    disabled={loading.analysis[fileId] || !selectedActivities[fileId] || selectedActivities[fileId].length === 0}
                  >
                    {loading.analysis[fileId] ? 'Analyzing...' : 'Run Analysis'}
                  </button>
                  <button
                    className="validate-button" // Nueva clase para el botón de validación
                    onClick={() => handleTemporalValidation(fileId)}
                    disabled={loading.validation[fileId] || !selectedActivities[fileId] || selectedActivities[fileId].length === 0}
                  >
                    {loading.validation[fileId] ? 'Validating...' : 'Temporal Validation'}
                  </button>
                </div>
              </div>
            );
          })}
        </section>
      </div>

      <section className="results-section">
        <h2>Analysis Results</h2>
        {selectedFileIds.length === 0 && <p className="info-text">Select and analyze a file to see results.</p>}
        {selectedFileIds.map(fileId => {
          const file = availableFiles.find(f => f.id === fileId);
          const resultData = analysisResults[fileId];
          const validationResult = temporalValidationData[fileId];

          // Solo mostrar PlotViewer si hay resultados de análisis Y NO hay error en el análisis
          const shouldShowPlotViewer = resultData && !resultData.error;
          // Solo mostrar TemporalValidationResults si hay datos de validación Y NO hay error en la validación
          const shouldShowValidation = validationResult && !validationResult.error;

          return (
            <div key={`results-${fileId}`} className="file-results-container">
              {/* Mostrar el nombre del archivo solo si hay resultados o datos de validación para mostrar */}
              {(shouldShowPlotViewer || shouldShowValidation) && <h3>Results for: {file?.name || fileId}</h3>}

              {loading.analysis[fileId] && <p>Loading analysis for {file?.name || fileId}...</p>}
              {loading.validation[fileId] && <p>Loading temporal validation for {file?.name || fileId}...</p>}
              
              {/* Mostrar resultados de validación temporal si existen */}
              {shouldShowValidation && (
                <TemporalValidationResults validationData={validationResult} />
              )}
              {validationResult && validationResult.error && (
                <div className="error-message">
                  <p>Temporal Validation Error for {file?.name || fileId}: {validationResult.error}</p>
                </div>
              )}

              {/* Mostrar PlotViewer si hay resultados de análisis */}
              {shouldShowPlotViewer && (
                <PlotViewer
                  fileName={file?.name || fileId}
                  plotData={resultData} // plotData es todo el objeto de resultado del análisis
                />
              )}
              {/* Mostrar error de análisis si existe */}
              {resultData && resultData.error && (
                 <div className="plot-viewer"> {/* Usar clase para consistencia de estilo */}
                    <h3 className="file-heading">{file?.name || fileId}</h3>
                    <div className="error-message">
                        <p>{resultData.error}</p>
                        {/* Opcionalmente, si tu backend devuelve plots incluso con error: */}
                        {/* resultData.plots?.events && <img src={`data:image/png;base64,${resultData.plots.events}`} alt="Events Timeline (Error Context)" /> */}
                    </div>
                 </div>
              )}
              {!loading.analysis[fileId] && !resultData && selectedFileIds.includes(fileId) && (
                <p className="info-text">Click "Run Analysis" for {file?.name || fileId} to see plots.</p>
              )}
            </div>
          );
        })}
      </section>
    </div>
  );
}

export default App;