import React, { useState, useEffect } from 'react';
import '../styles/components.css';

function AdvancedAnalysisViewer({ plotData, fileName }) {
  const [activeTab, setActiveTab] = useState('connectivity');
  const [expandedImage, setExpandedImage] = useState(null);

  // ✅ TODOS LOS HOOKS DEBEN IR AL PRINCIPIO, ANTES DE CUALQUIER RETURN

  // Debug received data
  useEffect(() => {
    console.log('🔬 AdvancedAnalysisViewer received:', {
      hasPlots: !!plotData?.plots,
      plotKeys: plotData?.plots ? Object.keys(plotData.plots) : [],
      hasMLResults: !!plotData?.ml_results,
      mlKeys: plotData?.ml_results ? Object.keys(plotData.ml_results) : []
    });
  }, [plotData]);

  // Preparar datos para el segundo useEffect
  const plots = plotData?.plots || {};
  const mlResults = plotData?.ml_results || {};
  const mlPlots = mlResults.plots || {};

  // ✅ SEGUNDO useEffect TAMBIÉN AL PRINCIPIO
  useEffect(() => {
    const tabs = [
      { 
        id: 'connectivity', 
        available: !!plots.connectivity
      },
      { 
        id: 'spectral', 
        available: !!plots.spectral_analysis
      },
      { 
        id: 'quality', 
        available: !!plots.signal_quality
      },
      { 
        id: 'ml_analysis', 
        available: !!(mlPlots.feature_importance || mlPlots.confusion_matrix || mlPlots.roc_curve)
      }
    ];

    const availableTabs = tabs.filter(tab => tab.available);
    if (availableTabs.length > 0 && !tabs.find(tab => tab.id === activeTab)?.available) {
      setActiveTab(availableTabs[0].id);
    }
  }, [plots.connectivity, plots.spectral_analysis, plots.signal_quality, mlPlots.feature_importance, mlPlots.confusion_matrix, mlPlots.roc_curve, activeTab]);

  // ✅ AHORA SÍ PODEMOS HACER EL RETURN CONDICIONAL
  if (!plotData?.plots && !plotData?.ml_results) {
    return (
      <div className="advanced-analysis-container">
        <div className="no-data-message">
          <p>🔬 Advanced analysis data not available</p>
          <p>Please run a complete analysis to see connectivity, spectral, and ML results.</p>
        </div>
      </div>
    );
  }

  // Definir tabs para el render
  const tabs = [
    { 
      id: 'connectivity', 
      label: 'Connectivity', 
      icon: '🔗',
      available: !!plots.connectivity
    },
    { 
      id: 'spectral', 
      label: 'Spectral Analysis', 
      icon: '📊',
      available: !!plots.spectral_analysis
    },
    { 
      id: 'quality', 
      label: 'Signal Quality', 
      icon: '📈',
      available: !!plots.signal_quality
    },
    { 
      id: 'ml_analysis', 
      label: 'ML Analysis', 
      icon: '🤖',
      available: !!(mlPlots.feature_importance || mlPlots.confusion_matrix || mlPlots.roc_curve)
    }
  ];

  const handleImageClick = (base64Image, title) => {
    setExpandedImage({ image: base64Image, title });
  };

  // Calcular availableTabs para el render
  const availableTabs = tabs.filter(tab => tab.available);

  return (
    <div className="advanced-analysis-container">
      <h3>Advanced NIRS Analysis - {fileName}</h3>
      
      <div className="analysis-tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''} ${!tab.available ? 'disabled' : ''}`}
            onClick={() => tab.available && setActiveTab(tab.id)}
            disabled={!tab.available}
          >
            <span className="tab-icon">{tab.icon}</span>
            {tab.label}
            {!tab.available && <span className="unavailable-badge">N/A</span>}
          </button>
        ))}
      </div>

      <div className="analysis-content">
        {activeTab === 'connectivity' && plots.connectivity && (
          <div className="analysis-section">
            <div className="section-header">
              <h4>Channel Connectivity Analysis</h4>
              <p className="section-description">
                This matrix shows the correlation between different NIRS channels. 
                High correlations may indicate shared physiological responses or artifacts.
              </p>
            </div>
            <div className="plot-container">
              <img
                src={`data:image/png;base64,${plots.connectivity}`}
                alt="Connectivity Matrix"
                className="analysis-plot"
                onClick={() => handleImageClick(plots.connectivity, 'Channel Connectivity Matrix')}
              />
            </div>
            <div className="analysis-insights">
              <h5>Interpretation Guidelines:</h5>
              <ul>
                <li><strong>Strong correlations (r &gt; 0.8):</strong> May indicate shared neural activity or systemic artifacts</li>
                <li><strong>Moderate correlations (0.5 &lt; r &lt; 0.8):</strong> Typical for nearby channels</li>
                <li><strong>Weak correlations (r &lt; 0.5):</strong> Independent signals, good for analysis</li>
              </ul>
            </div>
          </div>
        )}

        {activeTab === 'spectral' && plots.spectral_analysis && (
          <div className="analysis-section">
            <div className="section-header">
              <h4>Spectral Analysis</h4>
              <p className="section-description">
                Power spectral density analysis reveals the frequency components of NIRS signals.
                Different frequency bands correspond to different physiological processes.
              </p>
            </div>
            <div className="plot-container">
              <img
                src={`data:image/png;base64,${plots.spectral_analysis}`}
                alt="Spectral Analysis"
                className="analysis-plot"
                onClick={() => handleImageClick(plots.spectral_analysis, 'Spectral Analysis')}
              />
            </div>
            <div className="analysis-insights">
              <h5>Frequency Band Interpretation:</h5>
              <ul>
                <li><strong>0.01-0.08 Hz (Red):</strong> Systemic low-frequency oscillations</li>
                <li><strong>0.08-0.15 Hz (Green):</strong> Respiratory-related oscillations</li>
                <li><strong>0.15-0.4 Hz (Blue):</strong> Cardiac-related oscillations</li>
                <li><strong>Above 0.4 Hz:</strong> High-frequency noise and artifacts</li>
              </ul>
            </div>
          </div>
        )}

        {activeTab === 'quality' && plots.signal_quality && (
          <div className="analysis-section">
            <div className="section-header">
              <h4>Signal Quality Assessment</h4>
              <p className="section-description">
                Comprehensive analysis of signal quality across all channels to identify 
                potential artifacts and assess data reliability.
              </p>
            </div>
            <div className="plot-container">
              <img
                src={`data:image/png;base64,${plots.signal_quality}`}
                alt="Signal Quality"
                className="analysis-plot"
                onClick={() => handleImageClick(plots.signal_quality, 'Signal Quality Assessment')}
              />
            </div>
            <div className="analysis-insights">
              <h5>Quality Metrics:</h5>
              <ul>
                <li><strong>SNR &gt; 10 dB:</strong> Good signal quality</li>
                <li><strong>High correlation with mean:</strong> Potential systemic artifact</li>
                <li><strong>Extreme variance:</strong> May indicate motion artifacts</li>
                <li><strong>Consistent amplitude:</strong> Indicates stable optode contact</li>
              </ul>
            </div>
          </div>
        )}

        {activeTab === 'ml_analysis' && (
          <div className="analysis-section">
            <div className="section-header">
              <h4>Machine Learning Analysis</h4>
              <p className="section-description">
                Classification results and feature importance analysis from machine learning models.
              </p>
            </div>
            
            {/* ML Results Summary */}
            {mlResults.accuracy && (
              <div className="ml-summary">
                <h5>Classification Performance:</h5>
                <div className="ml-metrics">
                  <div className="metric">
                    <span className="metric-label">Best Classifier:</span>
                    <span className="metric-value">{mlResults.best_classifier || 'N/A'}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Accuracy:</span>
                    <span className="metric-value">{(mlResults.accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Cross-validation Score:</span>
                    <span className="metric-value">{(mlResults.cross_val_score * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            )}

            {/* ML Plots */}
            <div className="ml-plots-grid">
              {mlPlots.feature_importance && (
                <div className="plot-container">
                  <h5>Feature Importance</h5>
                  <img
                    src={`data:image/png;base64,${mlPlots.feature_importance}`}
                    alt="Feature Importance"
                    className="analysis-plot"
                    onClick={() => handleImageClick(mlPlots.feature_importance, 'Feature Importance')}
                  />
                </div>
              )}

              {mlPlots.confusion_matrix && (
                <div className="plot-container">
                  <h5>Confusion Matrix</h5>
                  <img
                    src={`data:image/png;base64,${mlPlots.confusion_matrix}`}
                    alt="Confusion Matrix"
                    className="analysis-plot"
                    onClick={() => handleImageClick(mlPlots.confusion_matrix, 'Confusion Matrix')}
                  />
                </div>
              )}

              {mlPlots.roc_curve && (
                <div className="plot-container">
                  <h5>ROC Curve</h5>
                  <img
                    src={`data:image/png;base64,${mlPlots.roc_curve}`}
                    alt="ROC Curve"
                    className="analysis-plot"
                    onClick={() => handleImageClick(mlPlots.roc_curve, 'ROC Curve')}
                  />
                </div>
              )}
            </div>

            {/* Top Features List */}
            {mlResults.top_features && mlResults.top_features.length > 0 && (
              <div className="analysis-insights">
                <h5>Top Discriminative Features:</h5>
                <ol className="top-features-list">
                  {mlResults.top_features.slice(0, 10).map((feature, index) => (
                    <li key={index} className={index === 0 ? "most-important" : ""}>
                      {feature}
                      {index === 0 && <span className="top-badge">HIGHEST IMPORTANCE</span>}
                    </li>
                  ))}
                </ol>
              </div>
            )}
          </div>
        )}

        {availableTabs.length === 0 && (
          <div className="no-data-message">
            <p>🚫 No advanced analysis data available</p>
            <p>This may occur if there was insufficient data or an error during processing.</p>
          </div>
        )}
      </div>

      {/* Modal para imagen expandida */}
      {expandedImage && (
        <div className="expanded-image-overlay" onClick={() => setExpandedImage(null)}>
          <div className="expanded-image-container">
            <div className="expanded-image-header">
              <h3>{expandedImage.title}</h3>
              <button className="close-button" onClick={() => setExpandedImage(null)}>×</button>
            </div>
            <img 
              src={`data:image/png;base64,${expandedImage.image}`}
              alt={expandedImage.title}
              className="expanded-image"
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default AdvancedAnalysisViewer;