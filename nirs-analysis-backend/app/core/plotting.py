import io
import base64
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
matplotlib.use('Agg')  # Use non-interactive backend

def convert_figure_to_base64(fig):
    """
    Convert a matplotlib figure to a base64 encoded string.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to convert.

    Returns:
    --------
    str
        Base64 encoded string of the figure.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

def plot_average_response(epochs, event_ids):
    """
    Plot the average NIRS response for each condition.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        The epoched NIRS data
    event_ids : dict
        Dictionary mapping event names to IDs
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for condition, event_id in event_ids.items():
        if condition in epochs.event_id:
            # Get data for this condition and average across epochs and channels
            data = epochs[condition].get_data()
            mean_data = np.mean(data, axis=(0, 1))  # Average across epochs and channels
            
            # Plot the average
            ax.plot(epochs.times, mean_data, label=condition)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Average NIRS Response by Condition')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)  # Mark stimulus onset
    
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_importance, feature_names, top_n=20):
    """
    Plot the importance of different features in the analysis.
    
    Parameters:
    -----------
    feature_importance : numpy.ndarray
        Array of feature importance values
    feature_names : list
        List of feature names
    top_n : int, optional
        Number of top features to display
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    # Ensure feature_importance and feature_names are numpy arrays
    feature_importance = np.array(feature_importance)
    feature_names = np.array(feature_names)
    
    # Get indices of top features
    if len(feature_importance) > top_n:
        indices = np.argsort(feature_importance)[-top_n:]
    else:
        indices = np.argsort(feature_importance)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(indices))
    ax.barh(y_pos, feature_importance[indices])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names[indices])
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top Features by Importance')
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix from classification results.
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    class_names : list
        List of class names
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use ConfusionMatrixDisplay from sklearn.metrics
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig

def generate_plots_for_api(analysis_result, activities):
    """
    Generate plots from analysis results and convert them to base64 encoded strings.
    """
    # If result already contains plots, just return them
    if 'plots' in analysis_result and isinstance(analysis_result['plots'], dict):
        return analysis_result['plots']
        
    # Otherwise, generate plots as before
    plots = {}
    
    # Create event_ids dictionary from activities list
    event_ids = {activity: i+1 for i, activity in enumerate(activities)}
    
    # Average response plot
    if 'epochs' in analysis_result:
        fig = plot_average_response(analysis_result['epochs'], event_ids)
        plots['average_response'] = convert_figure_to_base64(fig)
    
    # Feature importance plot
    if 'feature_importance' in analysis_result and 'feature_names' in analysis_result:
        fig = plot_feature_importance(
            analysis_result['feature_importance'],
            analysis_result['feature_names']
        )
        plots['feature_importance'] = convert_figure_to_base64(fig)
    
    # Error or diagnostic visualization
    if 'error' in analysis_result:
        if 'visualization' in analysis_result:
            plots['diagnostic'] = analysis_result['visualization']
            
    return plots