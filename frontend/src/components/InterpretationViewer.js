import React, { useState, useEffect } from 'react';
import '../styles/components.css';

/**
 * Brain region visualization component - simplified without highlighting
 */
const BrainRegionsImage = ({ region }) => {
  return (
    <div style={{ position: 'relative', width: '100%', maxWidth: '240px', height: '200px' }}>
      <img
        src="/assets/feature_importance.png" // Make sure this file exists in public/assets/
        alt="Brain Region Visualization"
        style={{ width: '100%', height: '100%', objectPosition: 'top' }}
      />
      {/* ✅ ELIMINADO: Todo el código del círculo SVG */}
    </div>
  );
};

/**
 * Helper function to safely render values that might be objects or strings
 */
function safeRenderValue(value) {
  if (typeof value === 'string') {
    return value;
  }
  if (typeof value === 'object' && value !== null) {
    // If it's an object, try to extract meaningful information
    if (value.function) return value.function;
    if (value.examples) return value.examples;
    if (value.anatomical_areas) {
      if (Array.isArray(value.anatomical_areas)) {
        return value.anatomical_areas.join(', ');
      }
      return String(value.anatomical_areas);
    }
    // Fallback: convert to JSON string
    return JSON.stringify(value);
  }
  // Fallback for other types
  return String(value || '-');
}

/**
 * Main interpretation viewer component with enhanced scientific explanations
 */
function InterpretationViewer({ interpretationData, topFeatures=[] }) {
    const [activeTab, setActiveTab] = useState('features');
    const [selectedFeature, setSelectedFeature] = useState(null);
    const [showHelpModal, setShowHelpModal] = useState(false);
    
    useEffect(() => {
      // Handle top features from ML analysis
      if (topFeatures && topFeatures.length > 0) {
        const topFeature = topFeatures[0];
        
        if (interpretationData?.feature_explanations?.[topFeature]) {
          setSelectedFeature(topFeature);
        } 
        else if (interpretationData && topFeature) {
          // If the top feature doesn't have an explanation yet, create one
          if (!interpretationData.feature_explanations) {
            interpretationData.feature_explanations = {};
          }
          
          const parts = topFeature.split('_');
          const region = parts[0] || 'prefrontal'; // Default region if parsing fails
          
          interpretationData.feature_explanations[topFeature] = {
            'region': region,
            'region_function': getRegionFunction(region),
            'measure_description': getMeasureDescription(topFeature),
            'wavelength_meaning': getWavelengthMeaning(parts[1] || '850')
          };
          
          setSelectedFeature(topFeature);
        }
      }
      else if (interpretationData?.feature_explanations) {
        // If no top features but we have explanations, select the first one
        const features = Object.keys(interpretationData.feature_explanations);
        if (features.length > 0) {
          setSelectedFeature(features[0]);
        }
        else {
          // Create a default explanation if none exists at all
          const defaultFeature = 'prefrontal_850_early_mean';
          
          interpretationData.feature_explanations[defaultFeature] = {
            'region': 'prefrontal',
            'region_function': 'Executive functions, working memory, and decision-making processes',
            'measure_description': 'Average activation during early response phase (1-4s post-stimulus)',
            'wavelength_meaning': '850nm wavelength - primarily sensitive to oxygenated hemoglobin (HbO)'
          };
          
          setSelectedFeature(defaultFeature);
        }
      }
    }, [interpretationData, topFeatures]); // Rerun when data or top features change

    if (!interpretationData) {
      return (
        <div className="interpretation-container">
          <h3 className="section-title">Neurophysiological Interpretation</h3>
          <p className="info-text">No interpretation data available. Please run the analysis first.</p>
        </div>
      );
    }

    const { 
      feature_explanations = {},
      event_descriptions = {},
      region_descriptions = {}
    } = interpretationData;

    // Sort features to show top features first
    const allFeatures = Object.keys(feature_explanations);
    const sortedFeatures = allFeatures.sort((a, b) => {
      // Top features first
      if (topFeatures.includes(a) && !topFeatures.includes(b)) return -1;
      if (!topFeatures.includes(a) && topFeatures.includes(b)) return 1;
      
      // Otherwise, sort by region name
      const regionA = a?.split('_')[0] || '';
      const regionB = b?.split('_')[0] || '';
      return regionA.localeCompare(regionB);
    });
  
    return (
      <div className="interpretation-container">
        <h3 className="section-title">Neurophysiological Interpretation</h3>
        
        {/* Tabs for different interpretation aspects */}
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'features' ? 'active' : ''}`}
            onClick={() => setActiveTab('features')}
          >
            Feature Analysis
          </button>
          <button 
            className={`tab ${activeTab === 'events' ? 'active' : ''}`}
            onClick={() => setActiveTab('events')}
          >
            Experimental Events
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
            Methodology
          </button>
        </div>
        
        {/* Content for the active tab */}
        <div className="tab-content">
          {/* Feature Analysis Tab */}
          {activeTab === 'features' && (
            <div className="features-panel">
              {/* List of features */}
              <div className="feature-list">
                <h4>Discriminative Features 
                  <button className="help-button" onClick={() => setShowHelpModal(true)}>
                    ? {/* Help icon */}
                  </button>
                </h4>
                
                {sortedFeatures.length > 0 ? (
                  <ul>
                      {/* Display top 15 features */}
                      {sortedFeatures.slice(0, 15).map((feature) => (
                        <li 
                          key={feature}
                          className={`feature-item ${selectedFeature === feature ? 'selected' : ''} ${topFeatures.includes(feature) ? 'top-feature' : ''}`}
                          onClick={() => setSelectedFeature(feature)}
                        >
                          <span className="feature-name">{formatFeatureName(feature)}</span>
                          {topFeatures.includes(feature) && <span className="top-badge">TOP</span>}
                        </li>
                      ))}
                      
                      {/* Show count if more features available */}
                      {sortedFeatures.length > 15 && (
                        <li className="feature-count">
                          ... and {sortedFeatures.length - 15} more features
                        </li>
                      )}
                  </ul>
                ) : (
                  <p className="info-text">No features available for analysis</p>
                )}
              </div>
              
              {/* Feature details */}
              {selectedFeature && feature_explanations[selectedFeature] && (
                <div className="feature-details">
                  <h4>{formatFeatureName(selectedFeature)}</h4>
                  <div className="detail-card">
                    {/* Brain region image - ✅ SIN CÍRCULO */}
                    <div className="brain-image-container">
                      <BrainRegionsImage 
                        region={feature_explanations[selectedFeature].region} 
                      />
                    </div>
                    {/* Textual explanation */}
                    <div className="explanation">
                      <p><strong>Brain Region:</strong> {capitalize(feature_explanations[selectedFeature].region || 'Unknown')}</p>
                      <p><strong>Neural Function:</strong> {safeRenderValue(feature_explanations[selectedFeature].region_function)}</p>
                      <p><strong>Measurement:</strong> {safeRenderValue(feature_explanations[selectedFeature].measure_description)}</p>
                      <p><strong>NIRS Signal:</strong> {safeRenderValue(feature_explanations[selectedFeature].wavelength_meaning)}</p>
                      <p><strong>Physiological Interpretation:</strong> {getPhysiologicalInterpretation(selectedFeature)}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* Experimental Events Tab */}
          {activeTab === 'events' && (
            <div className="events-panel">
              <h4>Experimental Task Descriptions</h4>
              <p className="explanation-text">Each event represents a distinct cognitive or motor task presented during the NIRS recording session.</p>
              {Object.keys(event_descriptions).length > 0 ? (
                <table className="event-table">
                  <thead>
                    <tr>
                      <th>Task Type</th>
                      <th>Neurocognitive Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(event_descriptions).map(([event, description]) => (
                      <tr key={event}>
                        <td>{safeRenderValue(event)}</td>
                        <td>{safeRenderValue(description)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="info-text">No experimental events data available.</p>
              )}
              <p className="note"><strong>Note:</strong> Task duration typically ranges from 10-30 seconds with inter-stimulus intervals of 15-45 seconds to allow hemodynamic response to return to baseline.</p>
            </div>
          )}
          
          {/* Brain Regions Tab */}
          {activeTab === 'regions' && (
            <div className="regions-panel">
              <h4>Functional Neuroanatomy</h4>
              <p className="explanation-text">NIRS channels record hemodynamic activity from these cortical regions, each associated with specific cognitive and motor functions.</p>
              {(() => {
                const enhancedRegions = getEnhancedRegionDescriptions(region_descriptions);
                const regionEntries = Object.entries(enhancedRegions);
                
                return regionEntries.length > 0 ? (
                  <table className="region-table">
                    <thead>
                      <tr>
                        <th>Cortical Region</th>
                        <th>Primary Functions</th>
                        <th>Role in Experimental Tasks</th>
                      </tr>
                    </thead>
                    <tbody>
                      {regionEntries.map(([region, details]) => (
                        <tr key={region}>
                          <td>{capitalize(region)}</td>
                          <td>{safeRenderValue(details?.function)}</td>
                          <td>{safeRenderValue(details?.examples)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p className="info-text">No brain regions data available.</p>
                );
              })()}
            </div>
          )}
          
          {/* Methodology Tab */}
          {activeTab === 'processing' && (
            <div className="processing-panel">
              <h4>Methodological Details</h4>
              <p className="explanation-text">Understanding the signal processing pipeline is essential for proper interpretation of NIRS results.</p>
              {/* Accordion for different methodology sections */}
              <div className="accordion">
                <div className="accordion-item">
                  <div className="accordion-header">
                    ✅ Signal Processing & Feature Extraction
                  </div>
                  <div className="accordion-content">
                    <ul>
                      <li><strong>Preprocessing:</strong> Optical density conversion, motion artifact detection, and baseline correction</li>
                      <li><strong>Temporal Filtering:</strong> Bandpass (0.01-0.5 Hz) to isolate hemodynamic response and remove physiological noise</li>
                      <li><strong>Spatial Filtering:</strong> Channel-based signal quality assessment and selection</li>
                      <li><strong>Feature Extraction:</strong> Amplitude, slope, and mean values across multiple time windows (early: 1-4s, middle: 5-10s, late: 11-15s)</li>
                      <li><strong>Feature Selection:</strong> F-score ranking to identify most discriminative signal components</li>
                    </ul>
                  </div>
                </div>
                <div className="accordion-item">
                  <div className="accordion-header">
                    📊 Machine Learning Analysis
                  </div>
                  <div className="accordion-content">
                    <ul>
                      <li><strong>Classification Models:</strong> SVM (Support Vector Machine), Random Forest, LDA (Linear Discriminant Analysis), and Ridge Classifier</li>
                      <li><strong>Validation:</strong> Block cross-validation to preserve temporal structure and prevent data leakage</li>
                      <li><strong>Parameter Tuning:</strong> Grid search optimization for each classifier</li>
                      <li><strong>Temporal Bias Check:</strong> Statistical tests against shuffled labels to ensure pattern robustness</li>
                    </ul>
                  </div>
                </div>
                <div className="accordion-item">
                  <div className="accordion-header">
                    ⚠️ Methodological Limitations
                  </div>
                  <div className="accordion-content">
                    <ul>
                      <li><strong>Spatial Resolution:</strong> Limited to cortical surface (1-3cm depth, ~1cm lateral resolution)</li>
                      <li><strong>Motion Artifacts:</strong> No advanced motion correction was applied</li>
                      <li><strong>Sample Size:</strong> Cross-validation with limited number of trials may affect generalization</li>
                      <li><strong>Signal Specificity:</strong> NIRS measures both neuronal and systemic vascular changes</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* Help modal with enhanced feature name explanation */}
        {showHelpModal && (
          <div className="modal-overlay" onClick={() => setShowHelpModal(false)}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h3>NIRS Feature Nomenclature</h3>
                <button className="close-button" onClick={() => setShowHelpModal(false)}>×</button>
              </div>
              <div className="modal-body">
                <p>NIRS features follow this naming structure:</p>
                <code>region_wavelength_timewindow_measure</code>
                <p><strong>Example:</strong> <code>prefrontal_850_early_mean</code></p>
                
                <h4>Components:</h4>
                <ul>
                  <li><strong>Region:</strong> Cortical area (prefrontal, motor, temporal, etc.)</li>
                  <li><strong>Wavelength:</strong> 
                    <ul>
                      <li>850nm: Primarily sensitive to oxygenated hemoglobin (HbO)</li>
                      <li>760nm: Primarily sensitive to deoxygenated hemoglobin (HbR)</li>
                    </ul>
                  </li>
                  <li><strong>Time window:</strong> 
                    <ul>
                      <li>early: 1-4 seconds post-stimulus</li>
                      <li>middle: 5-10 seconds post-stimulus</li>
                      <li>late: 11-15 seconds post-stimulus</li>
                    </ul>
                  </li>
                  <li><strong>Measure:</strong> Statistical property (mean, slope, std, peak, etc.)</li>
                </ul>
                
                <p><strong>Physiological meaning:</strong> These features capture different aspects of the hemodynamic response, which reflects neural activity through neurovascular coupling mechanisms.</p>
              </div>
            </div>
          </div>
        )}
      </div>
    );
}

/**
 * Format feature name for display
 */
function formatFeatureName(featureName) {
  if (!featureName) return 'Unknown Feature';
  
  // Split by underscores and capitalize each part
  const parts = featureName.split('_');
  const formatted = parts.map(part => capitalize(part)).join(' ');
  
  return formatted;
}

/**
 * Capitalize first letter of a string
 */
function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Get enhanced region descriptions with detailed functionality
 */
function getEnhancedRegionDescriptions(regionDescriptions) {
  const enhanced = {
    'prefrontal': {
      function: 'Executive control, working memory, attention regulation, decision-making',
      examples: 'Planning complex movements, inhibiting inappropriate responses, maintaining task goals'
    },
    'central_frontal': {
      function: 'Motor planning, cognitive control, response selection',
      examples: 'Preparation for voluntary movement, conflict monitoring, task switching'
    },
    'motor': {
      function: 'Primary motor execution, voluntary movement control',
      examples: 'Finger movements, hand coordination, motor sequence execution'
    },
    'lateral_frontal': {
      function: 'Language production, verbal working memory, cognitive flexibility',
      examples: 'Speech generation, verbal task processing, semantic retrieval'
    },
    'temporal': {
      function: 'Auditory processing, language comprehension, temporal sequence processing',
      examples: 'Sound discrimination, speech understanding, rhythm perception'
    },
    'parietal': {
      function: 'Spatial attention, sensorimotor integration, body awareness',
      examples: 'Spatial coordination, tactile processing, movement guidance'
    }
  };

  // Merge with backend descriptions if available
  const merged = { ...enhanced };
  
  // Safely handle regionDescriptions that might have complex objects
  if (regionDescriptions && typeof regionDescriptions === 'object') {
    Object.keys(regionDescriptions).forEach(region => {
      const description = regionDescriptions[region];
      
      if (merged[region]) {
        // If description is a string, use it as function
        if (typeof description === 'string') {
          merged[region].function = description;
        }
        // If description is an object, extract meaningful parts
        else if (typeof description === 'object' && description !== null) {
          if (description.function) {
            merged[region].function = safeRenderValue(description.function);
          }
          if (description.examples) {
            merged[region].examples = safeRenderValue(description.examples);
          }
        }
      } else {
        // Create new entry for unknown regions
        merged[region] = {
          function: safeRenderValue(description) || 'Cortical processing',
          examples: 'Various cognitive and motor tasks'
        };
      }
    });
  }

  return merged;
}

/**
 * Get region function based on brain region name
 */
function getRegionFunction(region) {
  const functions = {
    'prefrontal': 'Executive functions, working memory, attention control, and decision-making',
    'central_frontal': 'Motor planning, inhibition, and high-level cognitive control',
    'motor': 'Voluntary movement execution, motor sequence learning and control',
    'lateral_frontal': 'Language production, verbal working memory, and cognitive flexibility',
    'temporal': 'Auditory processing, language comprehension, and semantic memory',
    'parietal': 'Spatial processing, attention, and sensorimotor integration'
  };
  
  return functions[region] || 'Cortical processing related to cognitive and motor tasks';
}

/**
 * Get measure description based on feature name
 */
function getMeasureDescription(featureName) {
  const parts = featureName.split('_');
  const timeWindow = parts[2] || '';
  const measure = parts[3] || '';
  
  let description = '';
  
  if (timeWindow === 'early') {
    description = 'Early hemodynamic response (1-4s post-stimulus)';
  } else if (timeWindow === 'middle') {
    description = 'Peak hemodynamic response (5-10s post-stimulus)';
  } else if (timeWindow === 'late') {
    description = 'Late hemodynamic response (11-15s post-stimulus)';
  }
  
  if (measure === 'mean') {
    description += ' - average signal amplitude';
  } else if (measure === 'slope') {
    description += ' - rate of signal change';
  } else if (measure === 'peak') {
    description += ' - maximum signal amplitude';
  } else if (measure === 'std') {
    description += ' - signal variability';
  }
  
  return description || 'Hemodynamic response measurement';
}

/**
 * Get wavelength meaning
 */
function getWavelengthMeaning(wavelength) {
  if (wavelength === '850') {
    return '850nm wavelength - primarily sensitive to oxygenated hemoglobin (HbO), indicating increased neural activity';
  } else if (wavelength === '760') {
    return '760nm wavelength - primarily sensitive to deoxygenated hemoglobin (HbR), typically decreasing with neural activation';
  }
  return 'Near-infrared light wavelength for measuring hemodynamic changes';
}

/**
 * Generate physiological interpretation based on feature characteristics
 */
function getPhysiologicalInterpretation(featureName) {
  const parts = featureName.split('_');
  const region = parts[0];
  const isOxy = parts[1] === '850'; // Check if wavelength is 850nm (HbO)
  const timeWindow = parts[2] || '';
  
  let interpretation = '';
  
  if (isOxy) {
    // Interpretation for Oxygenated Hemoglobin (HbO)
    interpretation = `Increased oxygenated hemoglobin (HbO) concentration reflects heightened neural activity in the ${region} region`;
    if (timeWindow === 'early') {
      interpretation += ', indicating initial neural recruitment during task processing.';
    } else if (timeWindow === 'middle') {
      interpretation += ', corresponding to the peak of the hemodynamic response during sustained task engagement.';
    } else if (timeWindow === 'late') {
      interpretation += ', representing continued neural processing or return to baseline after task completion.';
    }
  } else {
    // Interpretation for Deoxygenated Hemoglobin (HbR)
    interpretation = `Changes in deoxygenated hemoglobin (HbR) concentration in the ${region} region`;
    interpretation += ' typically show an inverse relationship with neural activity (decreasing with activation).';
  }
  
  return interpretation;
}

export default InterpretationViewer;