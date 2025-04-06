"""
This module handles the visualization of NIRS data, generating plots
that can be sent to the frontend for display.
"""

import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def get_base64_encoded_figure(fig):
    """
    Converts a matplotlib figure to a base64 encoded string.
    
    Parameters:
    ----------
    fig : matplotlib.figure.Figure
        The figure to convert
        
    Returns:
    -------
    str
        Base64 encoded string representation of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

def plot_raw_data(raw_data):
    """
    Create a plot of the raw NIRS data.
    
    Parameters:
    ----------
    raw_data : mne.io.Raw
        The raw NIRS data
        
    Returns:
    -------
    str
        Base64 encoded plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))  # Increased height for more channels
    data, times = raw_data[:15, :]  # Show 15 channels instead of 10
    
    # Create a colormap for better channel differentiation
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    
    for i, d in enumerate(data):
        ax.plot(times, d + i*10, label=f"Channel {i}", color=colors[i], linewidth=1.0)
    
    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude + offset')
    ax.set_title('Raw NIRS Data (Multiple channels)')
    
    # Add legend on the right side
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()
    
    return get_base64_encoded_figure(fig)

def plot_average_response(epochs, event_ids):
    """
    Plot the average response for each condition.
    
    Parameters:
    ----------
    epochs : mne.Epochs
        The epoched data
    event_ids : dict
        Mapping of condition names to event IDs
        
    Returns:
    -------
    str
        Base64 encoded plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for condition, event_id in event_ids.items():
        if condition in epochs.event_id:
            data = epochs[condition].get_data().mean(axis=(0, 1))  # Average across epochs and channels
            ax.plot(epochs.times, data, label=condition)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Average NIRS Response by Condition')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    return get_base64_encoded_figure(fig)

def plot_pca_visualization(features, labels, event_ids):
    """
    Create a PCA plot showing the separation between different conditions.
    
    Parameters:
    ----------
    features : numpy.ndarray
        The feature matrix
    labels : numpy.ndarray
        The label vector
    event_ids : dict
        Mapping of condition names to event IDs
        
    Returns:
    -------
    str
        Base64 encoded plot
    """
    if len(features) < 2:
        return None
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get condition names from event IDs
    condition_names = {v: k for k, v in event_ids.items()}
    
    # Plot each condition
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            features_pca[mask, 0], 
            features_pca[mask, 1],
            label=condition_names[label],
            color=colors[i],
            alpha=0.7
        )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title('PCA Visualization of NIRS Patterns')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    return get_base64_encoded_figure(fig)