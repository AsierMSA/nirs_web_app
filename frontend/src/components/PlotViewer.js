import React, { useState } from 'react';
// Asegúrate de importar el CSS correcto que contiene .scrollable-plot-container etc.
import '../styles/components.css'; // O '../styles/PlotViewer.css' si ahí están los estilos

// ... (isEventsPlot, isChannelsPlot, isBrainVizDict functions remain the same) ...

function PlotViewer({ fileName, plotData }) {
  const [expandedImage, setExpandedImage] = useState(null);

  if (!plotData || Object.keys(plotData).length === 0) {
    return null; // Return null or some placeholder if no data
  }

  const hasError = plotData.error || plotData.message?.includes('error');
  const allPlots = plotData.plots || plotData; // Use plots if nested, otherwise assume top level
  const features = plotData.features || {}; // Handle features if present
  const interpretation = plotData.interpretation || {}; // Handle interpretation if present
  const temporalValidation = plotData.temporal_validation || null; // Handle temporal validation

  const handleImageClick = (base64Image, name) => {
    setExpandedImage({ src: base64Image, name: name });
  };

  const handleCloseExpanded = (e) => {
    e.stopPropagation();
    setExpandedImage(null);
  };

  // Helper function to check if a plot name indicates an events plot
  const isEventsPlot = (plotName) => {
    return plotName.toLowerCase().includes('event') &&
           !plotName.toLowerCase().includes('activation'); // Exclude activation plots
  };

  // Helper function to check if a plot name indicates a channels plot
  const isChannelsPlot = (plotName) => {
    return plotName.toLowerCase().includes('channel') &&
          !plotName.toLowerCase().includes('visualization'); // Exclude old activation plots if any
  };

  // Function to check for the brain visualizations dictionary
  const isBrainVizDict = (plotName) => {
    return plotName === 'brain_visualizations_by_event';
  };

  return (
    <div className="plot-viewer">
      <h3 className="file-heading">{fileName}</h3>

      {/* Display Temporal Validation Results if available */}
      {temporalValidation && (
        <div className="temporal-validation-results plot-item plot-item-medium"> {/* Added plot-item classes */}
          <h4>Temporal Bias Validation Results</h4>
          <p>Validation Score (AUC): {temporalValidation.validation_score?.toFixed(3) ?? 'N/A'}</p>
          <p>P-value: {temporalValidation.p_value?.toFixed(5) ?? 'N/A'}</p>
          <p>Conclusion: {temporalValidation.message ?? 'No conclusion available.'}</p>
          {temporalValidation.plot && (
            <> {/* Use Fragment to group */}
              <h5>Feature Performance vs. Time</h5>
              <div className="scrollable-plot-container">
                 <img
                   src={`data:image/png;base64,${temporalValidation.plot}`}
                   alt="Temporal Validation Plot"
                   className="plot-image"
                   style={{ maxWidth: 'none', height: 'auto', cursor: 'pointer' }} // Allow natural size, add cursor
                   onClick={() => handleImageClick(temporalValidation.plot, 'Temporal Validation Plot')}
                 />
              </div>
              <div className="scroll-hint">Scroll horizontally if the plot is too wide.</div>
            </>
          )}
        </div>
      )}

      {hasError && (
        <div className="error-message">
          <p>{plotData.error || plotData.message || "Analysis could not be completed"}</p>
          {/* Optionally show events plot even on error, using scrollable container */}
          {allPlots.events && (
             <div className="plot-item plot-item-large">
               <h4>Events Timeline (Error Context)</h4>
               {/* --- FIX: Use Scrollable Container --- */}
               <div className="scrollable-plot-container">
                 <img
                   src={`data:image/png;base64,${allPlots.events}`}
                   alt="Events Timeline"
                   className="plot-image"
                   // Let CSS handle sizing, only add essential inline styles
                   style={{ maxWidth: 'none', height: 'auto' }}
                 />
               </div>
               <div className="scroll-hint">Scroll horizontally to view the full timeline.</div>
               {/* --- END FIX --- */}
             </div>
           )}
        </div>
      )}

      {plotData.message && !hasError && !temporalValidation && (
        <div className="analysis-message">
          <p>{plotData.message}</p>
        </div>
      )}

      {!hasError && (
        <div className="plots-container">
          {/* Display events plots first and larger, using scrollable container */}
          {Object.entries(allPlots)
            .filter(([plotName, base64Image]) => base64Image && typeof base64Image === 'string' && isEventsPlot(plotName))
            .map(([plotName, base64Image]) => (
              <div key={plotName} className="plot-item plot-item-large">
                <h4>{formatPlotName(plotName)}</h4>
                {/* --- FIX: Use Scrollable Container --- */}
                <div className="scrollable-plot-container">
                  <img
                    src={`data:image/png;base64,${base64Image}`}
                    alt={`${plotName} plot`}
                    className="plot-image"
                    onClick={() => handleImageClick(base64Image, formatPlotName(plotName))}
                    // Let CSS handle sizing, only add essential inline styles
                    style={{ cursor: 'pointer', maxWidth: 'none', height: '400px' }}
                  />
                </div>
                <div className="scroll-hint">Scroll horizontally to view the full timeline.</div>
                {/* --- END FIX --- */}
              </div>
            ))}

          {/* Display channels plot with scrollable container */}
          {Object.entries(allPlots)
            .filter(([plotName, base64Image]) => base64Image && typeof base64Image === 'string' && isChannelsPlot(plotName))
            .map(([plotName, base64Image]) => (
              <div key={plotName} className="plot-item plot-item-large">
                <h4>{formatPlotName(plotName)}</h4>
                <div className="scrollable-channels-container">
                  <div className="image-container image-container-channels">
                    <img
                      src={`data:image/png;base64,${base64Image}`}
                      alt={`${plotName} plot`}
                      className="plot-image plot-image-channels"
                      onClick={() => handleImageClick(base64Image, formatPlotName(plotName))}
                      style={{ cursor: 'pointer' }} // Keep cursor pointer
                    />
                  </div>
                  <div className="scroll-hint">
                    Scroll down to see more channels
                  </div>
                </div>
              </div>
            ))}

          {/* Display Brain Visualizations per Event */}
          {Object.entries(allPlots)
            .filter(([plotName, data]) => isBrainVizDict(plotName) && data && typeof data === 'object')
            .flatMap(([_, eventPlots]) =>
              Object.entries(eventPlots)
                .filter(([eventName, base64Image]) => base64Image && typeof base64Image === 'string')
                .map(([eventName, base64Image]) => {
                  // Format title for brain maps
                  const plotTitle = formatPlotName(eventName.includes('vs') ? eventName : `Brain Activation (${eventName})`);
                  return (
                    <div key={`brain-viz-${eventName}`} className="plot-item plot-item-medium"> {/* Use medium size */}
                      <h4>{plotTitle}</h4>
                      <div className="image-container">
                        <img
                          src={`data:image/png;base64,${base64Image}`}
                          alt={plotTitle}
                          className="plot-image"
                          onClick={() => handleImageClick(base64Image, plotTitle)}
                          style={{ cursor: 'pointer' }}
                        />
                      </div>
                    </div>
                  );
                })
            )}

          {/* Render Feature Importance Plot if available */}
          {allPlots.feature_importance && features.top_features && (
            <div className="plot-item plot-item-medium"> {/* Or plot-item-large */}
              <h4>Feature Importance</h4>
               <div className="scrollable-plot-container">
                  <img
                    src={`data:image/png;base64,${allPlots.feature_importance}`}
                    alt="Feature Importance Plot"
                    className="plot-image"
                    style={{ maxWidth: '400px', height: '300px', cursor: 'pointer' }} // Allow natural size
                    onClick={() => handleImageClick(allPlots.feature_importance, 'Feature Importance')}
                  />
               </div>
               <div className="scroll-hint">Scroll horizontally if the plot is too wide.</div>
               <div className="top-features-list">
                 <h5>Top Discriminating Features:</h5>
                 <ul>
                   {features.top_features.slice(0, 10).map((feature, index) => (
                     <li key={index}>{feature}</li>
                   ))}
                 </ul>
               </div>
            </div>
          )}

          {/* Display other plots normally */}
          {Object.entries(allPlots)
            .filter(([plotName, data]) =>
                data && typeof data === 'string' &&
                !isEventsPlot(plotName) &&
                !isChannelsPlot(plotName) &&
                !isBrainVizDict(plotName) &&
                plotName !== 'feature_importance' &&
                // Add other specific plots handled elsewhere if needed
                plotName !== 'average_response' &&
                plotName !== 'confusion_matrix'
             )
            .map(([plotName, base64Image]) => (
              <div key={plotName} className="plot-item"> {/* Default size */}
                <h4>{formatPlotName(plotName)}</h4>
                <div className="image-container">
                  <img
                    src={`data:image/png;base64,${base64Image}`}
                    alt={`${plotName} plot`}
                    className="plot-image" // Let CSS handle max-width: 100%
                    onClick={() => handleImageClick(base64Image, formatPlotName(plotName))}
                    style={{ cursor: 'pointer' }}
                  />
                </div>
              </div>
            ))}

            {/* Render Average Response and Confusion Matrix (example placement) */}
             {allPlots.average_response && (
                <div key="average_response" className="plot-item plot-item-medium">
                    <h4>{formatPlotName('average_response')}</h4>
                    <div className="image-container">
                        <img src={`data:image/png;base64,${allPlots.average_response}`} alt="Average Response Plot" className="plot-image" onClick={() => handleImageClick(allPlots.average_response, formatPlotName('average_response'))} style={{ cursor: 'pointer' }} />
                    </div>
                </div>
             )}
             {allPlots.confusion_matrix && (
                <div key="confusion_matrix" className="plot-item plot-item-small">
                    <h4>{formatPlotName('confusion_matrix')}</h4>
                    <div className="image-container">
                        <img src={`data:image/png;base64,${allPlots.confusion_matrix}`} alt="Confusion Matrix" className="plot-image" onClick={() => handleImageClick(allPlots.confusion_matrix, formatPlotName('confusion_matrix'))} style={{ cursor: 'pointer' }} />
                    </div>
                </div>
             )}

        </div>
      )}

      {/* Interpretation Section */}
       {interpretation && Object.keys(interpretation).length > 0 && !hasError && (
         <div className="interpretation-section">
           <h4>Interpretation</h4>
           {interpretation.general && <p><strong>General:</strong> {interpretation.general}</p>}
           {interpretation.classification && <p><strong>Classification:</strong> {interpretation.classification}</p>}
           {interpretation.features && <p><strong>Features:</strong> {interpretation.features}</p>}
           {interpretation.comparison && <p><strong>Comparison:</strong> {interpretation.comparison}</p>}
         </div>
       )}


      {/* Modal for expanded image */}
      {expandedImage && (
        <div className="expanded-image-overlay" onClick={() => setExpandedImage(null)}>
          <div className="expanded-image-container">
            <div className="expanded-image-header">
              <h3>{expandedImage.name}</h3>
              <button className="close-button" onClick={handleCloseExpanded}>
                &times;
              </button>
            </div>
            <img
              src={`data:image/png;base64,${expandedImage.src}`}
              alt="Expanded view"
              className="expanded-image"
              onClick={(e) => e.stopPropagation()}
              style={{ maxWidth: '95%', maxHeight: '85vh', objectFit: 'contain' }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// Updated formatPlotName function (more robust)
function formatPlotName(name) {
  // Handle specific known keys first
  const knownNames = {
      'average_response': 'Average Hemodynamic Response',
      'confusion_matrix': 'Confusion Matrix',
      'feature_importance': 'Feature Importance',
      'events': 'Events Timeline',
      'channels': 'Channels Overview',
      'Left_vs_Right': 'Activation Difference (Left vs Right)' // Handle difference plot specifically
      // Add more specific mappings if needed
  };

  if (knownNames[name]) {
      return knownNames[name];
  }

  // Handle brain activation patterns like "Brain Activation (Rest)"
  if (name.startsWith('Brain Activation (')) {
      return name; // Assume already formatted
  }

  // General formatting for snake_case or camelCase
  return name
    .replace(/_/g, ' ') // Replace underscores with spaces
    .replace(/([A-Z])/g, ' $1') // Add space before capital letters (for camelCase)
    .replace(/(\d+)/g, ' $1') // Add space before numbers if needed
    .replace(/ vs /g, ' vs ') // Ensure ' vs ' keeps spaces
    .trim()
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1)) // Capitalize each word (handle lowercase correctly)
    .join(' ');
}


export default PlotViewer;