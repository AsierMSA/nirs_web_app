import React, { useState, useEffect } from 'react';
import { fetchFileActivities } from '../api/apiService';
import '../styles/components.css';

function ActivitySelector({ fileId, fileName, onSelectActivities }) {
  const [activities, setActivities] = useState([]);
  const [selectedActivities, setSelectedActivities] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadActivities = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const availableActivities = await fetchFileActivities(fileId);
        setActivities(availableActivities);
      } catch (err) {
        console.error(`Error loading activities for file ${fileId}:`, err);
        setError('Failed to load activities');
      } finally {
        setLoading(false);
      }
    };
    
    loadActivities();
  }, [fileId]);

  const handleActivityCheck = (activity) => {
    const isSelected = selectedActivities.includes(activity);
    
    let updatedActivities;
    if (isSelected) {
      updatedActivities = selectedActivities.filter(a => a !== activity);
    } else {
      updatedActivities = [...selectedActivities, activity];
    }
    
    setSelectedActivities(updatedActivities);
    onSelectActivities(updatedActivities);
  };

  if (loading) {
    return <div className="activity-loader">Loading activities...</div>;
  }

  if (error) {
    return <div className="activity-error">{error}</div>;
  }

  return (
    <div className="activity-selector">
      <h3 className="file-heading">{fileName}</h3>
      
      {activities.length === 0 ? (
        <p className="info-text">No activities found in this file</p>
      ) : (
        <div className="activity-list">
          {activities.map(activity => (
            <label key={activity} className="activity-item">
              <input
                type="checkbox"
                checked={selectedActivities.includes(activity)}
                onChange={() => handleActivityCheck(activity)}
              />
              <span className="activity-name">{activity}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

export default ActivitySelector;