import React from 'react';

const TemporalValidationResults = ({ validationData }) => {
  console.log("Temporal validation data:", validationData);
  if (!validationData) return null;

  const { p_value, shuffle_mean_accuracy, significant } = validationData;
  
  return (
    <div className="validation-results card mb-4">
      <div className="card-header bg-info text-white">
        <h5>Temporal Bias Validation Results</h5>
      </div>
      <div className="card-body">
        <div className="alert alert-info">
          <p><strong>What is this?</strong> This test checks if the machine learning model is detecting 
          true brain activity patterns or just exploiting the temporal structure of the experiment.</p>
        </div>
        
        <h6>Validation Test Results:</h6>
        <div className={`alert ${significant ? 'alert-success' : 'alert-warning'}`}>
          <p className="mb-1">
            <strong>Test Result:</strong> 
            {significant 
              ? ' PASSED ✓ - Results reflect true neural patterns' 
              : ' FAILED ⚠️ - Results may reflect temporal experiment structure'}
          </p>
          <p className="mb-1"><strong>p-value:</strong> {p_value.toFixed(4)}</p>
          <p className="mb-0"><strong>Random chance accuracy:</strong> {shuffle_mean_accuracy.toFixed(3)}</p>
        </div>
        
        <p className="small text-muted mt-3">
          A p-value below 0.05 indicates the model is learning true patterns, not just timing.
        </p>
      </div>
    </div>
  );
};

export default TemporalValidationResults;