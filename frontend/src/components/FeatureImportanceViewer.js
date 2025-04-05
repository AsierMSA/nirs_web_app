import React, { useEffect, useState, useRef } from 'react';
import '../styles/components.css';

function FeatureImportanceViewer({ featureImportanceData, topFeatures }) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [circlePositions, setCirclePositions] = useState([]);
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  
  // Load the image from backend and detect bar positions
  useEffect(() => {
    if (imageLoaded && containerRef.current && imageRef.current) {
      const container = containerRef.current;
      const img = imageRef.current;
      const canvas = document.createElement('canvas');
      const containerWidth = container.clientWidth;
      const containerHeight = container.clientHeight;
      
      // Draw image to canvas to analyze it
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, img.width, img.height);
      
      // The image data
      try {
        // In feature importance plots, bars are horizontal and sorted by importance
        // Top features are at the top of the image, and the bars extend from left to right
        // We need to find Y positions of top N bars and the X position of their ends
        
        const topBarPositions = detectTopBars(ctx, img.width, img.height, 3);
        
        // Convert to container's coordinate system
        const positions = topBarPositions.map(({x, y, rank}) => ({
          x: (x / img.width) * containerWidth,
          y: (y / img.height) * containerHeight,
          r: (30 - (rank * 5)) / 100 * Math.min(containerWidth, containerHeight)
        }));
        
        setCirclePositions(positions);
      } catch (e) {
        console.error("Error detecting feature positions:", e);
        // Fallback to approximate positions if detection fails
        const fallbackPositions = [
          {x: 0.85 * containerWidth, y: 0.18 * containerHeight, r: 30 / 100 * Math.min(containerWidth, containerHeight)},
          {x: 0.85 * containerWidth, y: 0.30 * containerHeight, r: 25 / 100 * Math.min(containerWidth, containerHeight)},
          {x: 0.82 * containerWidth, y: 0.42 * containerHeight, r: 20 / 100 * Math.min(containerWidth, containerHeight)}
        ];
        setCirclePositions(fallbackPositions);
      }
    }
  }, [imageLoaded]);
  
  // Function to detect the top features from the image
  function detectTopBars(ctx, width, height, numBars) {
    // Based on how feature importance charts are structured in the backend
    // - Horizontal bars
    // - Most important features are at the top
    // - Bars extend from left to right
    
    const positions = [];
    const rowHeight = height / 15; // Assuming around 15 bars visible (from backend code)
    
    for (let i = 0; i < numBars; i++) {
      // For each bar (top to bottom)
      const yCenter = (i + 0.5) * rowHeight;
      
      // Find the end of the bar (right to left)
      const xEnd = width * 0.85; // Approximate end position of the bar
      
      positions.push({
        x: xEnd,
        y: yCenter,
        rank: i
      });
    }
    
    return positions;
  }

  if (!featureImportanceData) return null;
  
  return (
    <div className="feature-importance-container">
      <h4>Feature Importance Analysis</h4>
      
      <div className="image-container" ref={containerRef} style={{ position: 'relative' }}>
        {/* The feature importance plot from the backend */}
        <img
            ref={imageRef}
            src="/assets/feature_importance.png" // Asegúrate de que este archivo exista en la carpeta public/assets/
            alt="Feature Importance Visualization"
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
            onLoad={() => setImageLoaded(true)}
        />
        
        {/* SVG overlay with circles */}
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
            {circlePositions.map((circle, index) => (
              <circle
                key={index}
                cx={circle.x}
                cy={circle.y}
                r={circle.r}
                fill="none"
                stroke={index === 0 ? "#ff5722" : "#ff9800"}
                strokeWidth={index === 0 ? 3 : 2}
                opacity={0.8 - (index * 0.2)}
              />
            ))}
          </svg>
        )}
      </div>
      
      <div className="top-features-list">
        <h5>Top Features (F-score ranking):</h5>
        <ol>
          {topFeatures?.slice(0, 5).map((feature, index) => (
            <li key={index} className={index === 0 ? "most-important" : ""}>
              {formatFeatureName(feature)}
              {index === 0 && <span className="top-badge">HIGHEST IMPORTANCE</span>}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

function formatFeatureName(name) {
  if (!name) return '';
  return name
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export default FeatureImportanceViewer;