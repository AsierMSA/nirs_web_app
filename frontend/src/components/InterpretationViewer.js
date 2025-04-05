import React, { useState, useEffect, useRef } from 'react';
import '../styles/components.css';

/**
 * Brain region visualization component that uses the feature importance image
 * instead of the SVG brain diagram
 */
const BrainRegionsImage = ({ region }) => {
  const [imageLoaded, setImageLoaded] = useState(false);
  const containerRef = useRef(null);
  const imageRef = useRef(null);
  const [highlightPosition, setHighlightPosition] = useState({ x: 0, y: 0 });
  
  // Determine highlight position based on region
  useEffect(() => {
    if (imageLoaded && containerRef.current) {
      const container = containerRef.current;
      const rect = container.getBoundingClientRect();
      
      // Map different positions based on region
      let position = { x: 0.5, y: 0.5 }; // default center
      
      if (region === 'prefrontal') {
        position = { x: 0.2, y: 0.2 };
      } else if (region === 'central_frontal') {
        position = { x: 0.5, y: 0.2 };
      } else if (region === 'lateral_frontal') {
        position = { x: 0.8, y: 0.2 };
      }
      
      setHighlightPosition({
        x: position.x * rect.width,
        y: position.y * rect.height,
        r: 30
      });
    }
  }, [region, imageLoaded]);
  
  return (
    <div ref={containerRef} style={{ position: 'relative', width: '100%', maxWidth: '240px', height: '200px' }}>
      {/* Use the feature importance image */}
      <img
        ref={imageRef}
        src="/assets/feature_importance.png"
        alt="Feature Importance Visualization"
        style={{ width: '100%', height: '100%', objectFit: 'contain' }}
        onLoad={() => setImageLoaded(true)}
      />
      
      {/* Add highlight circle */}
      {imageLoaded && (
        <svg 
          style={{ 
            position: 'absolute', 
            top: 0, 
            left: 0, 
            width: '100%', 
            height: '100%', 
            pointerEvents: 'none' 
          }}
        >
          <circle
            cx={highlightPosition.x}
            cy={highlightPosition.y}
            r={highlightPosition.r}
            fill="none"
            stroke="#ff5722"
            strokeWidth={3}
            opacity={0.8}
          />
        </svg>
      )}
    </div>
  );
};

/**
 * Main interpretation viewer component
 * Displays explanation of NIRS data including:
 * - Brain region functions
 * - Feature explanations
 * - Event descriptions
 * - Processing information
 */
function InterpretationViewer({ interpretationData, topFeatures=[] }) {
    const [activeTab, setActiveTab] = useState('features');
    const [selectedFeature, setSelectedFeature] = useState(null);
    const [showHelpModal, setShowHelpModal] = useState(false);
    
    useEffect(() => {
      console.log("InterpretationViewer - Received data:", { 
        hasInterpretation: !!interpretationData, 
        topFeatures, 
        featureExplanations: interpretationData?.feature_explanations ? Object.keys(interpretationData.feature_explanations) : 'none'
      });
      
      // Si tenemos características importantes del análisis ML, usar la primera (más importante)
      if (topFeatures && topFeatures.length > 0) {
        const topFeature = topFeatures[0];
        
        // Verificar si la característica más importante tiene una explicación
        if (interpretationData?.feature_explanations?.[topFeature]) {
          setSelectedFeature(topFeature);
          console.log("🔍 Using ML-selected top feature:", topFeature);
        } 
        // Si no hay explicación para la característica más importante, crearla
        else if (interpretationData && topFeature) {
          console.log("⚠️ Top feature missing explanation, will use default:", topFeature);
          
          // Asegurar que feature_explanations exista
          if (!interpretationData.feature_explanations) {
            interpretationData.feature_explanations = {};
          }
          
          // Extraer información del nombre de la característica
          const parts = topFeature.split('_');
          const region = parts[0] || 'prefrontal';
          const wavelength = parts[1] || '850';
          
          // Crear explicación por defecto
          interpretationData.feature_explanations[topFeature] = {
            'region': region,
            'region_function': interpretationData?.region_descriptions?.[region]?.function || 
                              'Executive functions and cognitive processing',
            'measure_description': 'Most important feature from machine learning analysis',
            'wavelength_meaning': wavelength === '850' ? 
              '850nm - primarily sensitive to oxygenated hemoglobin' : 
              '760nm - primarily sensitive to deoxygenated hemoglobin'
          };
          
          setSelectedFeature(topFeature);
        }
      }
      // Fallback al comportamiento anterior
      else if (interpretationData && interpretationData.feature_explanations) {
        const features = Object.keys(interpretationData.feature_explanations);
        if (features.length > 0) {
          setSelectedFeature(features[0]);
          console.log("⚠️ Fallback to first available feature:", features[0]);
        }
        // Si no hay características, crear una por defecto
        else {
          const defaultFeature = 'prefrontal_850_mean';
          console.log("⚠️ No features available, creating default:", defaultFeature);
          
          interpretationData.feature_explanations[defaultFeature] = {
            'region': 'prefrontal',
            'region_function': 'Executive functions and cognitive processing',
            'measure_description': 'Average activation level',
            'wavelength_meaning': '850nm - primarily sensitive to oxygenated hemoglobin'
          };
          
          setSelectedFeature(defaultFeature);
        }
      }
    }, [interpretationData, topFeatures]);
    
    // Si no hay datos de interpretación, mostrar un mensaje en lugar de no renderizar nada
    if (!interpretationData) {
      return (
        <div className="interpretation-container">
          <h3 className="section-title">Results Interpretation</h3>
          <p>No interpretation data available. Analysis in progress...</p>
        </div>
      );
    }
    
    // Asegurarse de que estos objetos siempre existan para evitar errores
    const region_descriptions = interpretationData.region_descriptions || {};
    const feature_explanations = interpretationData.feature_explanations || {};
    const event_descriptions = interpretationData.event_descriptions || {};
    
    // Sort features by region for better organization
    const sortedFeatures = Object.keys(feature_explanations).sort((a, b) => {
      // First put top features at the top
      if (topFeatures.includes(a) && !topFeatures.includes(b)) return -1;
      if (!topFeatures.includes(a) && topFeatures.includes(b)) return 1;
      
      // Then sort by region
      const regionA = a?.split('_')[0] || '';
      const regionB = b?.split('_')[0] || '';
      return regionA.localeCompare(regionB);
    });
  
  return (
    <div className="interpretation-container">
      <h3 className="section-title">Results Interpretation</h3>
      
      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'features' ? 'active' : ''}`}
          onClick={() => setActiveTab('features')}
        >
          Feature Meanings
        </button>
        <button 
          className={`tab ${activeTab === 'events' ? 'active' : ''}`}
          onClick={() => setActiveTab('events')}
        >
          Event Labels
        </button>
        <button 
          className={`tab ${activeTab === 'regions' ? 'active' : ''}`}
          onClick={() => setActiveTab('regions')}
        >
          Brain Regions
        </button>
        <button 
          className={`tab ${activeTab === 'processing' ? 'active' : ''}`}
          onClick={() => setActiveTab('processing')}
        >
          Processing Info
        </button>
      </div>
      
      <div className="tab-content">
        {activeTab === 'features' && (
          <div className="features-panel">
            <div className="feature-list">
              <h4>Top Features 
                <button className="help-button" onClick={() => setShowHelpModal(true)}>
                  ?
                </button>
              </h4>
              
              {/* Show features list or a message if none are available */}
              {sortedFeatures.length > 0 ? (
                <ul>
                    {sortedFeatures.slice(0, 15).map(feature => (
                    <li 
                        key={feature}
                        className={`
                        ${selectedFeature === feature ? 'selected' : ''} 
                        ${topFeatures.length > 0 && feature === topFeatures[0] ? 'most-important' : ''}
                        `}
                        onClick={() => setSelectedFeature(feature)}
                    >
                        {formatFeatureName(feature)}
                        {topFeatures.length > 0 && feature === topFeatures[0] && (
                        <span className="top-badge">HIGHEST F-SCORE</span>
                        )}
                    </li>
                    ))}
                </ul>
                ) : (
                <p>No feature data available</p>
                )}
            </div>
            
            {/* Feature details with null safety checks */}
            {selectedFeature && feature_explanations[selectedFeature] && (
              <div className="feature-details">
                <h4>{formatFeatureName(selectedFeature)}</h4>
                <div className="detail-card">
                  <div className="brain-image-container">
                    <BrainRegionsImage 
                      region={feature_explanations[selectedFeature].region} 
                    />
                    {/* Remove div.highlight as we now use SVG for highlighting */}
                  </div>
                  <div className="explanation">
                    <p><strong>Region:</strong> {capitalize(feature_explanations[selectedFeature].region || 'Unknown')} lobe</p>
                    <p><strong>Function:</strong> {feature_explanations[selectedFeature].region_function || 'Unknown'}</p>
                    <p><strong>Measurement:</strong> {feature_explanations[selectedFeature].measure_description || 'Unknown'}</p>
                    <p><strong>Wavelength:</strong> {feature_explanations[selectedFeature].wavelength_meaning || 'Unknown'}</p>
                    <p><strong>Physiological meaning:</strong> Changes in blood oxygenation during cognitive/motor tasks</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        
        {activeTab === 'events' && (
          <div className="events-panel">
            <h4>Event Descriptions</h4>
            <table className="event-table">
              <thead>
                <tr>
                  <th>Event Label</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(event_descriptions).map(([event, description]) => (
                  <tr key={event}>
                    <td>{event}</td>
                    <td>{description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        
        {activeTab === 'regions' && (
          <div className="regions-panel">
            <h4>Brain Regions & Functions</h4>
            <table className="region-table">
              <thead>
                <tr>
                  <th>Region</th>
                  <th>Function</th>
                  <th>Examples in Experiment</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(region_descriptions).map(([region, details]) => (
                  <tr key={region}>
                    <td>{capitalize(region)}</td>
                    <td>{details.function || '-'}</td>
                    <td>{details.examples || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        
        {/* Processing info tab (no changes needed) */}
        {activeTab === 'processing' && (
          <div className="processing-panel">
            <h4>Data Processing Information</h4>
            <div className="accordion">
              <div className="accordion-item">
                <div className="accordion-header">
                  ✅ Preprocessing Steps Applied
                </div>
                <div className="accordion-content">
                  <ul>
                    <li>Bandpass filtering (0.01-0.5 Hz)</li>
                    <li>Channel selection</li>
                    <li>Feature extraction</li>
                    <li>Feature selection using F-score</li>
                  </ul>
                </div>
              </div>
              <div className="accordion-item">
                <div className="accordion-header">
                  ⚠️ Processing Limitations
                </div>
                <div className="accordion-content">
                  <ul>
                    <li>No motion artifact correction was applied</li>
                    <li>Limited spatial resolution</li>
                    <li>Cross-validation with limited sample size</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Help modal with feature name explanation */}
      {showHelpModal && (
        <div className="modal-overlay" onClick={() => setShowHelpModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>How to Read Feature Names</h3>
              <button className="close-button" onClick={() => setShowHelpModal(false)}>×</button>
            </div>
            <div className="modal-body">
              <p>Feature names have the following structure:</p>
              <code>region_wavelength_measure</code>
              <ul>
                <li><strong>region:</strong> Brain area (prefrontal, central_frontal, lateral_frontal)</li>
                <li><strong>wavelength:</strong> Light wavelength used (760nm or 850nm)</li>
                <li><strong>measure:</strong> Type of measurement extracted from the signal</li>
              </ul>
              <p>Example: <code>prefrontal_850_early_mean</code> = Average signal in prefrontal region during early response phase, measuring oxygenated hemoglobin.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Format a feature name for display by capitalizing each word
 */
function formatFeatureName(name) {
  if (!name) return '';
  return name
    .split('_')
    .map(word => capitalize(word))
    .join(' ');
}

/**
 * Capitalize the first letter of a string
 */
function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export default InterpretationViewer;