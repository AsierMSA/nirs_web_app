import React, { useState } from 'react';
import '../styles/components.css';

function PlotViewer({ fileName, plotData }) {
  // State to track which image is expanded (null means no image is expanded)
  const [expandedImage, setExpandedImage] = useState(null);
  
  if (!plotData || Object.keys(plotData).length === 0) {
    return null;
  }

  // Function to handle image click to expand/collapse
  const handleImageClick = (base64Image, plotName) => {
    if (expandedImage && expandedImage.src === base64Image) {
      // If clicking the currently expanded image, collapse it
      setExpandedImage(null);
    } else {
      // Otherwise expand the clicked image
      setExpandedImage({ src: base64Image, name: plotName });
    }
  };

  // If there's an error message, display it with any diagnostic plots
  const hasError = plotData.error || plotData.message?.includes('error');
  
  // Function to determine if a plot is an events plot
  const isEventsPlot = (plotName) => {
    return plotName.toLowerCase().includes('event') || 
           plotName.toLowerCase() === 'events_plot' ||
           plotName.toLowerCase().includes('timeline');
  };
  
  // Function to determine if a plot is a channels plot
  const isChannelsPlot = (plotName) => {
    return plotName.toLowerCase().includes('channel') &&
          !plotName.toLowerCase().includes('visualization');
  };
  
  return (
    <div className="plot-viewer">
      <h3 className="file-heading">{fileName}</h3>
      
      {hasError && (
        <div className="error-message">
          <p>{plotData.error || "Analysis could not be completed"}</p>
        </div>
      )}
      
      {plotData.message && !hasError && (
        <div className="analysis-message">
          <p>{plotData.message}</p>
        </div>
      )}
      
      <div className="plots-container">
        {/* Display events plots first and larger */}
        {Object.entries(plotData.plots || {})
          .filter(([plotName]) => isEventsPlot(plotName))
          .map(([plotName, base64Image]) => (
            <div key={plotName} className="plot-item plot-item-large">
              <h4>{formatPlotName(plotName)}</h4>
              <div className="image-container image-container-large">
                <img 
                  src={`data:image/png;base64,${base64Image}`} 
                  alt={`${plotName} plot`}
                  className="plot-image"
                  onClick={() => handleImageClick(base64Image, formatPlotName(plotName))}
                  style={{ cursor: 'pointer', width: '100%', maxHeight: '500px', objectFit: 'contain' }}
                />
              </div>
            </div>
          ))}
        
        {/* Display channels plot with scrollable container */}
        {Object.entries(plotData.plots || {})
          .filter(([plotName]) => isChannelsPlot(plotName))
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
                    style={{ cursor: 'pointer' }}
                  />
                </div>
                <div className="scroll-hint">
                  Scroll down to see more channels
                </div>
              </div>
            </div>
          ))}
        
        {/* Display other plots normally */}
        {Object.entries(plotData.plots || {})
          .filter(([plotName]) => !isEventsPlot(plotName) && !isChannelsPlot(plotName))
          .map(([plotName, base64Image]) => (
            <div key={plotName} className="plot-item">
              <h4>{formatPlotName(plotName)}</h4>
              <div className="image-container">
                <img 
                  src={`data:image/png;base64,${base64Image}`} 
                  alt={`${plotName} plot`}
                  className="plot-image"
                  onClick={() => handleImageClick(base64Image, formatPlotName(plotName))}
                  style={{ cursor: 'pointer' }}
                />
              </div>
            </div>
          ))}
        
        {/* Handle case where plots aren't nested in a plots object */}
        {!plotData.plots && Object.entries(plotData).filter(([key, value]) => 
          typeof value === 'string' && value.startsWith('iVBORw')
        ).map(([plotName, base64Image]) => (
          <div key={plotName} className={`plot-item ${isEventsPlot(plotName) ? 'plot-item-large' : ''}`}>
            <h4>{formatPlotName(plotName)}</h4>
            <div className={`image-container ${isEventsPlot(plotName) ? 'image-container-large' : ''}`}>
              <img 
                src={`data:image/png;base64,${base64Image}`} 
                alt={`${plotName} plot`}
                className="plot-image"
                onClick={() => handleImageClick(base64Image, formatPlotName(plotName))}
                style={{ 
                  cursor: 'pointer',
                  ...(isEventsPlot(plotName) ? { width: '100%', maxHeight: '500px', objectFit: 'contain' } : {})
                }}
              />
            </div>
          </div>
        ))}
      </div>
      
      {/* Modal for expanded image */}
      {expandedImage && (
        <div className="expanded-image-overlay" onClick={() => setExpandedImage(null)}>
          <div className="expanded-image-container">
            <div className="expanded-image-header">
              <h3>{expandedImage.name}</h3>
              <button className="close-button" onClick={(e) => {
                e.stopPropagation();
                setExpandedImage(null);
              }}>
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

function formatPlotName(name) {
  // Convert snake_case to Title Case
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export default PlotViewer;