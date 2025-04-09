import React, { useState, useEffect, useRef } from 'react';
import '../styles/components.css';

/**
 * Brain region visualization component with enhanced highlighting
 */
const BrainRegionsImage = ({ region }) => {
  const [imageLoaded, setImageLoaded] = useState(false);
  const containerRef = useRef(null);
  const imageRef = useRef(null);
  const [highlightPosition, setHighlightPosition] = useState({ x: 0, y: 0 });
  
  // Better region mapping with more precise positioning
  useEffect(() => {
    if (imageLoaded && containerRef.current) {
      const container = containerRef.current;
      const rect = container.getBoundingClientRect();
      
      // Improved positioning map for brain regions
      let position = { x: 0.5, y: 0.5 }; // default center
      
      if (region === 'prefrontal') {
        position = { x: 0.2, y: 0.2 };
      } else if (region === 'central_frontal') {
        position = { x: 0.5, y: 0.2 };
      } else if (region === 'lateral_frontal') {
        position = { x: 0.8, y: 0.2 };
      } else if (region === 'motor') {
        position = { x: 0.5, y: 0.4 };
      } else if (region === 'temporal') {
        position = { x: 0.8, y: 0.5 };
      } else if (region === 'parietal') {
        position = { x: 0.5, y: 0.6 };
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
      <img
        ref={imageRef}
        src="/assets/feature_importance.png" 
        alt="Brain Region Visualization"
        style={{ width: '100%', height: '100%',objectPosition: 'top' }}
        onLoad={() => setImageLoaded(true)}
      />
      
      {imageLoaded && (
        <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
          <circle
            cx={highlightPosition.x}
            cy={highlightPosition.y}
            r={highlightPosition.r}
            fill="rgba(255, 87, 34, 0.2)"
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
          if (!interpretationData.feature_explanations) {
            interpretationData.feature_explanations = {};
          }
          
          const parts = topFeature.split('_');
          const region = parts[0] || 'prefrontal';
          const wavelength = parts[1] || '850';
          const timeWindow = parts[2] || 'early';
          
          // Enhanced feature explanation with scientific context
          interpretationData.feature_explanations[topFeature] = {
            'region': region,
            'region_function': getRegionFunction(region),
            'measure_description': getMeasureDescription(timeWindow, parts[3] || 'mean'),
            'wavelength_meaning': wavelength === '850' ? 
              '850nm wavelength - primarily sensitive to oxygenated hemoglobin (HbO)' : 
              '760nm wavelength - primarily sensitive to deoxygenated hemoglobin (HbR)'
          };
          
          setSelectedFeature(topFeature);
        }
      }
      else if (interpretationData && interpretationData.feature_explanations) {
        const features = Object.keys(interpretationData.feature_explanations);
        if (features.length > 0) {
          setSelectedFeature(features[0]);
        }
        else {
          // Create default explanation if none exists
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
    }, [interpretationData, topFeatures]);
    
    if (!interpretationData) {
      return (
        <div className="interpretation-container">
          <h3 className="section-title">Results Interpretation</h3>
          <p>Analysis in progress. Results will be available once processing is complete.</p>
        </div>
      );
    }
    
    const region_descriptions = interpretationData.region_descriptions || {};
    const feature_explanations = interpretationData.feature_explanations || {};
    const event_descriptions = interpretationData.event_descriptions || createDefaultEventDescriptions();
    
    // Sort features by importance then region
    const sortedFeatures = Object.keys(feature_explanations).sort((a, b) => {
      if (topFeatures.includes(a) && !topFeatures.includes(b)) return -1;
      if (!topFeatures.includes(a) && topFeatures.includes(b)) return 1;
      
      const regionA = a?.split('_')[0] || '';
      const regionB = b?.split('_')[0] || '';
      return regionA.localeCompare(regionB);
    });
  
    return (
      <div className="interpretation-container">
        <h3 className="section-title">Neurophysiological Interpretation</h3>
        
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
        
        <div className="tab-content">
          {activeTab === 'features' && (
            <div className="features-panel">
              <div className="feature-list">
                <h4>Discriminative Features 
                  <button className="help-button" onClick={() => setShowHelpModal(true)}>
                    ?
                  </button>
                </h4>
                
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
                            <span className="top-badge">HIGHEST DISCRIMINATIVE POWER</span>
                          )}
                      </li>
                      ))}
                  </ul>
                  ) : (
                  <p>No feature data available</p>
                  )}
              </div>
              
              {selectedFeature && feature_explanations[selectedFeature] && (
                <div className="feature-details">
                  <h4>{formatFeatureName(selectedFeature)}</h4>
                  <div className="detail-card">
                    <div className="brain-image-container">
                      <BrainRegionsImage 
                        region={feature_explanations[selectedFeature].region} 
                      />
                    </div>
                    <div className="explanation">
                      <p><strong>Brain Region:</strong> {capitalize(feature_explanations[selectedFeature].region || 'Unknown')}</p>
                      <p><strong>Neural Function:</strong> {feature_explanations[selectedFeature].region_function || 'Unknown'}</p>
                      <p><strong>Measurement:</strong> {feature_explanations[selectedFeature].measure_description || 'Unknown'}</p>
                      <p><strong>NIRS Signal:</strong> {feature_explanations[selectedFeature].wavelength_meaning || 'Unknown'}</p>
                      <p><strong>Physiological Interpretation:</strong> {getPhysiologicalInterpretation(selectedFeature)}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          
          {activeTab === 'events' && (
            <div className="events-panel">
              <h4>Experimental Task Descriptions</h4>
              <p className="explanation-text">Each event represents a distinct cognitive or motor task presented during the NIRS recording session.</p>
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
                      <td>{event}</td>
                      <td>{description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="note"><strong>Note:</strong> Task duration typically ranges from 10-30 seconds with inter-stimulus intervals of 15-45 seconds to allow hemodynamic response to return to baseline.</p>
            </div>
          )}
          
          {activeTab === 'regions' && (
            <div className="regions-panel">
              <h4>Functional Neuroanatomy</h4>
              <p className="explanation-text">NIRS channels record hemodynamic activity from these cortical regions, each associated with specific cognitive and motor functions.</p>
              <table className="region-table">
                <thead>
                  <tr>
                    <th>Cortical Region</th>
                    <th>Primary Functions</th>
                    <th>Role in Experimental Tasks</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(getEnhancedRegionDescriptions(region_descriptions)).map(([region, details]) => (
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
          
          {activeTab === 'processing' && (
            <div className="processing-panel">
              <h4>Methodological Details</h4>
              <p className="explanation-text">Understanding the signal processing pipeline is essential for proper interpretation of NIRS results.</p>
              <div className="accordion">
                <div className="accordion-item">
                  <div className="accordion-header">
                    ✅ Signal Processing & Feature Extraction
                  </div>
                  <div className="accordion-content">
                    <ul>
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
 * Get description for different measure and time window combinations
 */
function getMeasureDescription(timeWindow, measureType) {
  let timeDescription = '';
  
  switch(timeWindow) {
    case 'early':
      timeDescription = 'early response phase (1-4s post-stimulus)';
      break;
    case 'middle':
      timeDescription = 'middle response phase (5-10s post-stimulus)';
      break;
    case 'late':
      timeDescription = 'late response phase (11-15s post-stimulus)';
      break;
    default:
      timeDescription = 'response period';
  }
  
  const measures = {
    'mean': `Average activation during ${timeDescription}`,
    'slope': `Rate of change in hemodynamic response during ${timeDescription}`,
    'peak': `Maximum amplitude of response during ${timeDescription}`,
    'std': `Variability of hemodynamic signal during ${timeDescription}`
  };
  
  return measures[measureType] || `Measurement of hemodynamic activity during ${timeDescription}`;
}

/**
 * Generate physiological interpretation based on feature characteristics
 */
function getPhysiologicalInterpretation(featureName) {
  const parts = featureName.split('_');
  const region = parts[0];
  const isOxy = parts[1] === '850';
  const timeWindow = parts[2] || '';
  
  let interpretation = '';
  
  if (isOxy) {
    interpretation = `Increased oxygenated hemoglobin (HbO) concentration reflects heightened neural activity in the ${region} region`;
    if (timeWindow === 'early') {
      interpretation += ', indicating initial neural recruitment during task processing.';
    } else if (timeWindow === 'middle') {
      interpretation += ', corresponding to the peak of the hemodynamic response during sustained task engagement.';
    } else if (timeWindow === 'late') {
      interpretation += ', representing continued neural processing or return to baseline after task completion.';
    }
  } else {
    interpretation = `Changes in deoxygenated hemoglobin (HbR) concentration in the ${region} region`;
    interpretation += ' typically show an inverse relationship with neural activity (decreasing with activation).';
  }
  
  return interpretation;
}

/**
 * Create default event descriptions if none are provided
 */
function createDefaultEventDescriptions() {
  return {
    'Finger Sequencing': 'Sequential finger tapping task that engages fine motor control and motor sequence learning circuits primarily in motor and premotor cortices',
    'Simple Tapping': 'Basic rhythmic finger tapping that activates primary motor cortex with minimal cognitive load',
    'Motor Imagery': 'Mental simulation of movement without physical execution, engaging similar neural circuits as actual movement but with reduced primary motor activation',
    'Rest': 'Baseline condition with no specific task demands, used as reference for hemodynamic response analysis',
    'Working Memory': 'Mental manipulation and temporary storage of information, primarily engaging prefrontal and parietal cortical networks',
    'Bimanual Coordination': 'Coordinated movement of both hands, requiring interhemispheric communication and increased cognitive-motor integration'
  };
}

/**
 * Enhance region descriptions with more scientific detail
 */
function getEnhancedRegionDescriptions(existingDescriptions) {
  const defaultDescriptions = {
    'prefrontal': {
      function: 'Executive function, working memory, decision-making, cognitive control',
      examples: 'Active during complex problem-solving, task-switching, and inhibition tasks'
    },
    'motor': {
      function: 'Movement planning and execution, motor sequence learning',
      examples: 'Primary activation during finger tapping and coordination tasks'
    },
    'central_frontal': {
      function: 'Cognitive control, response inhibition, action selection',
      examples: 'Engaged during complex motor sequences and tasks requiring controlled responses'
    },
    'temporal': {
      function: 'Auditory processing, language comprehension, semantic processing',
      examples: 'Activated during verbal instructions and auditory stimuli processing'
    },
    'parietal': {
      function: 'Spatial awareness, attention, sensorimotor integration',
      examples: 'Important for visuomotor coordination and spatial aspects of tasks'
    }
  };
  
  // Merge existing descriptions with defaults
  const enhanced = {...defaultDescriptions};
  Object.entries(existingDescriptions).forEach(([region, details]) => {
    if (!enhanced[region]) {
      enhanced[region] = details;
    } else {
      enhanced[region] = {
        ...enhanced[region],
        function: details.function || enhanced[region].function,
        examples: details.examples || enhanced[region].examples
      };
    }
  });
  
  return enhanced;
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