"""
This module contains functions for processing NIRS data.
It includes methods for loading NIRS files, visualizing events and signals,
and extracting basic features from raw data.
"""

import os
from matplotlib import patches
import mne
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import traceback

def map_numeric_annotations_to_descriptive(raw_data, annotation_map=None):
    """
    Map numeric annotations in a raw NIRS file to descriptive labels.
    
    Parameters:
    ----------
    raw_data : mne.io.Raw
        Raw NIRS data with annotations
    annotation_map : dict
        Mapping from numeric codes to descriptive labels
        e.g., {'1': 'Hand_Movement', '2': 'Rest', '3': 'Imagination'}
        
    Returns:
    -------
    mne.io.Raw
        Raw data with updated annotations
    """
    if annotation_map is None:
        # Default mapping based on common schemes
        annotation_map = {
            '1': 'Exercise',
            '2': 'Rest',
            '3': 'Imagination',
            '0': 'Baseline'
        }
    
    # Create new annotations with the same parameters but mapped descriptions
    new_annotations = mne.Annotations(
        onset=[a['onset'] for a in raw_data.annotations],
        duration=[a['duration'] for a in raw_data.annotations],
        description=[annotation_map.get(a['description'], a['description']) 
                     for a in raw_data.annotations]
    )
    
    # Set the new annotations
    raw_data.set_annotations(new_annotations)
    
    return raw_data

def load_nirs_data(file_path):
    """
    Load NIRS data from a specified file path.

    Parameters:
    ----------
    file_path : str
        The path to the NIRS data file.

    Returns:
    -------
    mne.io.Raw
        The loaded NIRS data.
    """
    try:
        raw_data = mne.io.read_raw_fif(file_path, preload=True)
        print_available_channels(raw_data)  # Añade esta línea
        return raw_data
    except Exception as e:
        print(f"Error loading NIRS data: {e}")
        return None

def encode_figure_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def extract_events_and_visualize(raw_data, event_ids, tmin=-5.0, tmax=20.0):
    """
    Extract events from raw data and create visualization.
    
    Parameters:
    ----------
    raw_data : mne.io.Raw
        Raw NIRS data
    event_ids : dict
        Dictionary mapping event descriptions to numeric IDs
    tmin : float
        Start time relative to event onset for data extraction
    tmax : float
        End time relative to event onset for data extraction
        
    Returns:
    -------
    dict
        Dictionary containing event information and visualizations
    """
    # Extract valid events
    valid_events = []
    event_info = []
    max_time = raw_data.times[-1]
    sfreq = raw_data.info['sfreq']
    
    # Check if there are any annotations at all
    if len(raw_data.annotations) == 0:
        print("WARNING: No annotations found in the file!")
        return {
            'error': 'No events/annotations found in the file',
            'message': 'The NIRS file does not contain any event markers or annotations.',
            'total_duration': max_time
        }
    
    # Print all available annotations for debugging
    print(f"All annotations in file ({len(raw_data.annotations)}):")
    for i, annot in enumerate(raw_data.annotations[:20]):
        print(f"  {i+1}. '{annot['description']}' at {annot['onset']:.1f}s (duration: {annot['duration']:.1f}s)")
    if len(raw_data.annotations) > 20:
        print(f"  ... and {len(raw_data.annotations) - 20} more")
    
    # Count annotations per description
    descriptions_count = {}
    for annot in raw_data.annotations:
        desc = annot['description']
        descriptions_count[desc] = descriptions_count.get(desc, 0) + 1
    
    # Check which requested activities are actually in the annotations
    found_activities = [act for act in event_ids if any(annot['description'] == act for annot in raw_data.annotations)]
    missing_activities = [act for act in event_ids if act not in found_activities]
    
    if missing_activities:
        print(f"\nWARNING: The following requested activities were not found in annotations:")
        for act in missing_activities:
            print(f"  - {act}")
    
    # Extract valid events for the requested activities
    for annot in raw_data.annotations:
        # Convert NumPy string to Python string before using as dict key
        description = str(annot['description'])
        
        if description in event_ids:
            onset = annot['onset']
            
            # Validate event position
            if onset < abs(tmin):
                continue
            if (onset + tmax) > max_time:
                print(f"Event at {onset:.1f}s exceeds recording duration")
                continue
            
            # Convert onset time to sample index
            onset_sample = int(onset * sfreq)
            valid_events.append([onset_sample, 0, event_ids[description]])  # Use the Python string
            
            # Store event info for later
            event_info.append({
                'onset': onset,
                'description': description,  # Store Python string
                'duration': annot['duration'],
                'code': event_ids[description]  # Use Python string
            })
    
    
    if not valid_events:
        if found_activities:
            return {
                'error': 'No valid events found within time boundaries',
                'message': f'Found {len(found_activities)} matching activities, but none have sufficient data.',
                'found_activities': found_activities,
                'missing_activities': missing_activities
            }
        else:
            return {
                'error': 'No matching events found',
                'message': f'None of the requested activities were found in the file annotations.',
                'available_annotations': list(descriptions_count.keys())
            }
    
    # Generate diagnostic visualization of events
    fig_events = generate_events_plot(raw_data, valid_events, event_info, event_ids, sfreq, max_time)
    events_plot = encode_figure_to_base64(fig_events)
    
    # Add event statistics to return data
    event_types = sorted(list(set([e['description'] for e in event_info])))
    event_stats = {
        'total_events': len(valid_events),
        'events_by_type': {desc: len([e for e in event_info if e['description'] == desc]) for desc in event_types},
        'time_span': [min([e['onset'] for e in event_info]), max([e['onset'] for e in event_info])]
    }
    
    return {
        'valid_events': valid_events,
        'event_info': event_info,
        'event_stats': event_stats,
        'events_plot': events_plot,
        'sfreq': sfreq,
        'max_time': max_time,
        'event_types': event_types
    }

def generate_events_plot(raw_data, valid_events, event_info, event_ids, sfreq, max_time):
    """Generate detailed visualization of events"""
    fig_events = plt.figure(figsize=(15, 8))
    
    # First plot: Raw data visualization with event markers
    ax1 = plt.subplot(2, 1, 1)
    
    # Determine optimal plot parameters
    n_channels_to_show = min(5, len(raw_data.ch_names))
    ch_names_to_show = raw_data.ch_names[:n_channels_to_show]
    
    # Determine optimal plot duration based on event density
    event_times = [event[0] / sfreq for event in valid_events]
    if event_times:
        # Find region with highest event density
        window_size = 60  # seconds
        max_density_start = 0
        if len(event_times) > 10:
            max_density = 0
            for start in range(0, int(max_time) - window_size + 1, 5):
                end = start + window_size
                events_in_window = sum(1 for t in event_times if start <= t < end)
                density = events_in_window / window_size
                if density > max_density:
                    max_density = density
                    max_density_start = start
        
        plot_duration = min(60, max_time)
        plot_start_time = max(0, min(max_density_start, max_time - plot_duration))
    else:
        plot_duration = min(60, max_time)
        plot_start_time = 0
    
    # Extract data for plotting
    start_sample = int(plot_start_time * sfreq)
    end_sample = int((plot_start_time + plot_duration) * sfreq)
    data, times = raw_data[:n_channels_to_show, start_sample:end_sample]
    times = times + plot_start_time
    
    # Plot the data lines
    for i, ch_data in enumerate(data):
        # Normalize for better visualization
        ch_data = ch_data - np.mean(ch_data)
        ch_data = ch_data / (np.std(ch_data) if np.std(ch_data) > 0 else 1)
        # Plot with offset for visibility
        ax1.plot(times, ch_data + i*3, linewidth=0.5)
    
    # Add channel labels and customize axes
    ax1.set_yticks(np.arange(0, n_channels_to_show*3, 3))
    ax1.set_yticklabels(ch_names_to_show)
    
    # Mark events on the plot
    max_labels = 20
    visible_events = [event for event in valid_events 
                     if plot_start_time <= event[0]/sfreq <= plot_start_time + plot_duration]
    
    if len(visible_events) > max_labels:
        events_to_label = visible_events[::len(visible_events)//max_labels + 1][:max_labels]
        for event in visible_events:
            event_time = event[0] / sfreq
            ax1.axvline(event_time, color='r', linestyle='--', alpha=0.4)
    else:
        events_to_label = visible_events
    
    for event in events_to_label:
        event_time = event[0] / sfreq
        event_code = event[2]
        event_desc = next((k for k, v in event_ids.items() if v == event_code), f"Code {event_code}")
        ax1.axvline(event_time, color='r', linestyle='--', alpha=0.7)
        
        mins = int(event_time) // 60
        secs = event_time % 60
        time_str = f"{mins:02d}:{secs:04.1f}"
        
        ax1.text(event_time, n_channels_to_show*3, f"{event_desc}\n({time_str})", 
                rotation=90, verticalalignment='bottom', fontsize=8)
    
    # Update plot appearance
    ax1.set_xlabel('Time (s)')
    ax1.set_title(f'Raw Data with Event Markers ({len(visible_events)} events)')
    ax1.set_xlim(plot_start_time, plot_start_time + plot_duration)
    
    if plot_start_time > 0 or plot_start_time + plot_duration < max_time:
        ax1.text(0.5, 0.01, 
                f"Showing {plot_start_time:.1f}s - {plot_start_time + plot_duration:.1f}s of {max_time:.1f}s total",
                transform=ax1.transAxes, ha='center', fontsize=8, style='italic')
    
    # Second plot: timeline showing events with labels
    ax2 = plt.subplot(2, 1, 2)
    
    # Create color map for different event types
    event_types = sorted(list(set([e['description'] for e in event_info])))
    colors = plt.cm.tab10(np.linspace(0, 1, len(event_types)))
    color_map = {event_type: colors[i] for i, event_type in enumerate(event_types)}
    
    # Determine visible time window for second plot
    timeline_start = plot_start_time
    timeline_end = plot_start_time + plot_duration
    
    # Filter and plot events
    visible_events_timeline = [e for e in event_info 
                              if (timeline_start <= e['onset'] <= timeline_end) or 
                                 (e['onset'] < timeline_start and e['onset'] + e['duration'] > timeline_start)]
    
    # Calculate max events to label based on available space
    max_event_labels = 15
    
    if len(visible_events_timeline) > max_event_labels:
        step = max(1, len(visible_events_timeline) // max_event_labels)
        events_to_label = visible_events_timeline[::step]
    else:
        events_to_label = visible_events_timeline
    
    # Plot events as colored spans
    for event in event_info:
        desc = event['description']
        onset = event['onset']
        duration = event['duration'] if event['duration'] > 0 else 5.0
        
        if (onset >= timeline_start and onset <= timeline_end) or \
           (onset < timeline_start and onset + duration > timeline_start):
            ax2.axvspan(onset, onset + duration, alpha=0.3, color=color_map[desc])
    
    # Add labels for selected events
    for event in events_to_label:
        desc = event['description']
        onset = event['onset'] 
        duration = event['duration'] if event['duration'] > 0 else 5.0
        
        label_x = max(timeline_start, min(timeline_end, onset + duration/2))
        
        if len(events_to_label) <= max_event_labels:
            ax2.text(label_x, 0.5, desc, 
                     horizontalalignment='center', 
                     verticalalignment='center',
                     rotation=90 if duration < 10 else 0,
                     fontsize=8, color='black', 
                     transform=ax2.get_xaxis_transform())
    
    # Customize timeline plot
    ax2.set_xlim(timeline_start, timeline_end) 
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time (s)')
    ax2.set_yticks([])
    ax2.set_title(f'Event Timeline ({len(visible_events_timeline)} events in view)')
    ax2.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Create a legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[t], alpha=0.3) for t in event_types]
    ax2.legend(handles, event_types, loc='upper right', fontsize=8)
    
    # Add indication of shown time range if not showing all data
    if timeline_start > 0 or timeline_end < max_time:
        ax2.text(0.5, 0.01, 
                f"Showing {timeline_start:.1f}s - {timeline_end:.1f}s of {max_time:.1f}s total",
                transform=ax2.transAxes, ha='center', fontsize=8, style='italic')
    
    plt.tight_layout(pad=1.5)
    return fig_events

def extract_features_from_events(raw_data, valid_events, event_ids, tmin=-5.0, tmax=20.0):
    """
    Extract features from events in raw data using all available channels directly.
    """
    # Variables to store results
    data_chunks = []
    labels = []
    condition_names = []
    feature_names = []
    X_features = []
    sfreq = raw_data.info['sfreq']
    
    # Calculate start and end samples for the time window
    start_offset = int(tmin * sfreq)
    end_offset = int(tmax * sfreq)
    
    # For visualization by condition
    condition_data = {cond: [] for cond in event_ids}
    
    # Extract data for each event
    print(f"Processing {len(valid_events)} valid events...")
    for i, event in enumerate(valid_events):
        onset_sample = event[0]
        event_code = event[2]
        condition = list(event_ids.keys())[list(event_ids.values()).index(event_code)]
        
        # Calculate start and end samples for this event
        start = onset_sample + start_offset
        end = onset_sample + end_offset
        
        # Extract data for this time window
        data = raw_data.get_data(start=start, stop=end)
        
        # Store for later
        data_chunks.append(data)
        labels.append(event_code)
        condition_names.append(condition)
        condition_data[condition].append(data)
    
    # Generate average response by condition plot
    fig_avg, ax = plt.subplots(figsize=(12, 7))
    time_points = np.linspace(tmin, tmax, end_offset - start_offset)
    
    for condition, condition_chunks in condition_data.items():
        if condition_chunks:
            # Average across events and channels for this condition
            avg_data = np.mean(np.array(condition_chunks), axis=0).mean(axis=0)
            ax.plot(time_points, avg_data, linewidth=2, label=condition)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Average Response by Condition')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.tight_layout()
    avg_response_plot = encode_figure_to_base64(fig_avg)
    
    # Extract unique channel identifiers
    unique_channels = []
    for ch_name in raw_data.ch_names:
        parts = ch_name.split(' ')
        if len(parts) >= 1:
            channel = parts[0]  # Extraer solo el identificador S*_D*
            if channel not in unique_channels:
                unique_channels.append(channel)
    
    # Ordenar canales para mejor visualización
    unique_channels.sort()
    
    # Generate channel analysis plot
    fig_channels, axs = plt.subplots(min(10, len(unique_channels)), 1, figsize=(15, 4*min(10, len(unique_channels))))
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    
    region_data = {}
    
    # Solo mostramos los primeros canales para no saturar el gráfico
    channels_to_show = unique_channels[:10]
    
    for i, channel in enumerate(channels_to_show):
        region_data[channel] = {}
        ax = axs[i]
        
        # Identify indexes for this channel
        channel_picks = [idx for idx, ch in enumerate(raw_data.ch_names) if channel in ch]
        
        if channel_picks:
            for condition, condition_chunks in condition_data.items():
                if condition_chunks:
                    # Extract channel data for all events in this condition
                    channel_condition_data = [chunk[channel_picks, :] for chunk in condition_chunks]
                    
                    if channel_condition_data:
                        # Average across events
                        channel_avg = np.mean([np.mean(chunk, axis=0) for chunk in channel_condition_data], axis=0)
                        ax.plot(time_points, channel_avg, linewidth=2, label=condition)
                        
                        # Store for results
                        region_data[channel][condition] = {
                            'mean': float(np.mean(channel_avg)),
                            'peak': float(np.max(channel_avg)),
                            'std': float(np.std(channel_avg))
                        }
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Channel {channel}')
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    channels_plot = encode_figure_to_base64(fig_channels)
    
    # Feature extraction
    # Define time windows for different phases
    baseline_indices = (time_points >= -5) & (time_points < 0)
    early_indices = (time_points >= 1) & (time_points <= 4)
    middle_indices = (time_points >= 5) & (time_points <= 10)
    late_indices = (time_points >= 11) & (time_points <= 15)
    
    # Identify available wavelengths
    wavelengths = []
    for ch in raw_data.ch_names:
        parts = ch.split(' ')
        if len(parts) > 1 and parts[-1].isdigit():
            wavelengths.append(int(parts[-1]))
    
    unique_wavelengths = np.unique(wavelengths) if wavelengths else []
    
    # For each data chunk (event)
    for i, data_chunk in enumerate(data_chunks):
        features = []
        
        # For each wavelength
        wave_groups = {}
        if len(unique_wavelengths) >= 2:
            for wl in unique_wavelengths:
                wave_picks = [idx for idx, ch in enumerate(raw_data.ch_names) if ch.endswith(f" {wl}")]
                wave_groups[wl] = wave_picks
        else:
            wave_groups['all'] = list(range(len(raw_data.ch_names)))
        
        # For each channel
        for channel in unique_channels:
            for wave_label, wave_picks in wave_groups.items():
                channel_wave_picks = [idx for idx in wave_picks if channel in raw_data.ch_names[idx]]
                
                if not channel_wave_picks:
                    # Skip channels with no data for this wavelength
                    continue
                
                # Feature name prefix
                feature_prefix = f"{channel}_{wave_label}"
                
                # Extract data for these channels
                data = data_chunk[channel_wave_picks, :]
                
                try:
                    # Calculate features
                    baseline_mean = data[:, baseline_indices].mean(axis=1).mean() if np.any(baseline_indices) else 0
                    
                    # Early window mean
                    early_mean = data[:, early_indices].mean(axis=1).mean() if np.any(early_indices) else 0
                    features.append(early_mean)
                    if i == 0:
                        feature_names.append(f"{feature_prefix}_early_mean")
                    
                    # Middle window mean
                    middle_mean = data[:, middle_indices].mean(axis=1).mean() if np.any(middle_indices) else 0
                    features.append(middle_mean)
                    if i == 0:
                        feature_names.append(f"{feature_prefix}_middle_mean")
                    
                    # Late window mean
                    late_mean = data[:, late_indices].mean(axis=1).mean() if np.any(late_indices) else 0
                    features.append(late_mean)
                    if i == 0:
                        feature_names.append(f"{feature_prefix}_late_mean")
                    
                    # Early slope
                    slope_early = (middle_mean - early_mean) / 5.0 if middle_mean != early_mean else 0
                    features.append(slope_early)
                    if i == 0:
                        feature_names.append(f"{feature_prefix}_slope_early")
                    
                    # Late slope
                    slope_late = (late_mean - middle_mean) / 5.0 if late_mean != middle_mean else 0
                    features.append(slope_late)
                    if i == 0:
                        feature_names.append(f"{feature_prefix}_slope_late")
                    
                    # Amplitude relative to baseline
                    peak_mean = max(early_mean, middle_mean, late_mean)
                    amplitude = peak_mean - baseline_mean
                    features.append(amplitude)
                    if i == 0:
                        feature_names.append(f"{feature_prefix}_amplitude")
                    
                    # Overall variability
                    total_std = np.std(data) if data.size > 0 else 0
                    features.append(total_std)
                    if i == 0:
                        feature_names.append(f"{feature_prefix}_std")
                    
                except Exception as e:
                    print(f"Error in feature {feature_prefix}: {e}")
        
        # Handle NaN values
        features = np.array(features)
        if np.any(np.isnan(features)):
            features = np.nan_to_num(features, nan=0.0)
        
        X_features.append(features)
    
    # Convert to numpy arrays
    X_features = np.array(X_features)
    labels = np.array(labels)
    
    return {
        'X_features': X_features, 
        'feature_names': feature_names,
        'labels': labels,
        'condition_names': condition_names,
        'time_points': time_points,
        'plots': {
            'average_response': avg_response_plot,
            'channels': channels_plot
        },
        'region_data': region_data,
        'n_events': len(valid_events),
        'event_ids': event_ids
    }
def print_available_channels(raw_data):
    """
    Print available channels and extract source-detector pairs.
    """
    print("\n===== AVAILABLE CHANNELS =====")
    print(f"Total channels: {len(raw_data.ch_names)}")
    
    # Print all channels
    print("\nComplete channels:")
    for i, ch in enumerate(raw_data.ch_names):
        print(f"  {i}: {ch}")
    
    # Extract and count unique source-detector pairs
    source_detector_pairs = set()
    for ch in raw_data.ch_names:
        # Extract S-D pairs (different possible formats)
        if 'S' in ch and 'D' in ch:
            parts = ch.split()
            sd_pair = parts[0] if ' ' in ch else ch.split('_')[0]
            source_detector_pairs.add(sd_pair)
    
    # Print unique source-detector pairs
    print("\nUnique source-detector pairs:")
    for i, pair in enumerate(sorted(source_detector_pairs)):
        print(f"  {i}: {pair}")
    
    print("\n==============================\n")
    
    return source_detector_pairs
def extract_features_from_raw(raw_data, activities):
    """Extract features from raw NIRS data"""
    # Create event_ids dictionary from activity list
    event_ids = {activity: i+1 for i, activity in enumerate(activities)}
    
    # First extract events
    events_result = extract_events_and_visualize(raw_data, event_ids)
    
    # Check if there was an error in event extraction
    if 'error' in events_result:
        return events_result
    
    # Then extract features from events
    features_result = extract_features_from_events(
        raw_data, 
        events_result['valid_events'],
        event_ids
    )
    
    return features_result
def analyze_nirs_file(file_path, activities, annotation_map=None):
    try:
        # Import ML functions here to avoid circular imports
        from .nirs_ml import apply_machine_learning
        
        raw_data = load_nirs_data(file_path)
        if raw_data is not None:
            # If no specific activities requested, use all non-boundary annotations
            if not activities:
                annotations = set([a['description'] for a in raw_data.annotations 
                                  if not a['description'].endswith('boundary')])
                activities = list(annotations)
            
            # Apply annotation mapping if provided
            if annotation_map:
                raw_data = map_numeric_annotations_to_descriptive(raw_data, annotation_map)
                
                # Update activity list with mapped names if needed
                mapped_activities = []
                for activity in activities:
                    if activity in annotation_map:
                        mapped_activities.append(annotation_map[activity])
                    else:
                        mapped_activities.append(activity)
                activities = mapped_activities
            
            # Create event_ids dictionary from activity list
            event_ids = {activity: i+1 for i, activity in enumerate(activities)}
            
            # Extract events and create visualizations
            events_result = extract_events_and_visualize(raw_data, event_ids)
            if 'error' in events_result:
                return events_result
                
            # Extract features from events
            features_result = extract_features_from_events(
                raw_data, 
                events_result['valid_events'],
                event_ids
            )

            # Channel visualization
            activations = calculate_activations(raw_data)
            channels_visualization = create_brain_visualization(raw_data, activations)

            # Get interpretation metadata
            from .nirs_ml import generate_interpretation_metadata
            interpretation_data = generate_interpretation_metadata(
                features_result['feature_names'],
                raw_data,
                None  # We pass None instead of brain_regions
            )
            
            # Apply machine learning if we have enough data
            if features_result['X_features'].shape[0] > 2 and len(np.unique(features_result['labels'])) > 1:
                from .nirs_ml import validate_against_temporal_bias

                temporal_validation = apply_machine_learning(
                features_result['X_features'], 
                features_result['labels'],
                features_result['feature_names']
            )
                print(f"[DEBUG] Top features received in analyze_nirs_file: {temporal_validation.get('top_features', [])}")
                
                # Combine results
                combined_results = {
                    **events_result['event_stats'],
                    'features': {
                        'shape': features_result['X_features'].shape,
                        'feature_count': len(features_result['feature_names']),
                        'top_features': temporal_validation.get('top_features', [])
                    },
                    'plots': {
                        'events': events_result['events_plot'],
                        'average_response': features_result['plots']['average_response'],
                        'channels_visualization': channels_visualization,
                        'channels': features_result['plots']['channels'],  # New channel visualization
                        'classifier_comparison': temporal_validation.get('plots', {}).get('classifier_comparison'),
                        'confusion_matrix': temporal_validation.get('plots', {}).get('confusion_matrix'),
                        'feature_importance': temporal_validation.get('plots', {}).get('feature_importance'),
                        'learning_curve': temporal_validation.get('plots', {}).get('learning_curve')
                    },
                    'channel_data': features_result['region_data'],  # Keep name for compatibility
                    'best_classifier': temporal_validation.get('best_classifier'),
                    'interpretation': interpretation_data,
                    'accuracy': temporal_validation.get('accuracy'),
                    'ml_params': temporal_validation.get('params', {})
                }
            else:
                # Not enough data for ML analysis
                combined_results = {
                    **events_result['event_stats'],
                    'features': {
                        'shape': features_result['X_features'].shape,
                        'feature_count': len(features_result['feature_names']),
                    },
                    'plots': {
                        'events': events_result['events_plot'],
                        'average_response': features_result['plots']['average_response'],
                        'channels_visualization': channels_visualization,
                        'channels': features_result['plots']['channels']  # New channel visualization
                    },
                    'channel_data': features_result['region_data'],  # Keep name for compatibility
                    'interpretation': interpretation_data,
                    'warning': 'Insufficient data for machine learning analysis'
                }
                
            return combined_results
        else:
            return {'error': 'Failed to load NIRS data.'}
    except Exception as e:
        return {
            'error': f'Analysis failed: {str(e)}',
            'traceback': traceback.format_exc()
        }
def create_brain_visualization(raw_data, activations=None):
    """
    2D visualization of NIRS channels with activation levels and head silhouette
    
    Parameters:
    ----------
    raw_data : mne.io.Raw
        Loaded NIRS data
    activations : dict, optional
        Dictionary {channel_name: activation_value} (0-1)
        
    Returns:
    -------
    str
        Base64 encoded image or None if failed
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib import patches
    from io import BytesIO
    import numpy as np
    import matplotlib.pyplot as plt
    import base64
    import traceback

    plt.switch_backend('Agg')

    try:
        # Get original channel positions
        ch_positions = np.array([ch['loc'][:2] for ch in raw_data.info['chs']])
        
        # Create figure manually
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Initial configuration
        ax.set_title('NIRS Channels Activation Map', fontsize=14, pad=15)
        ax.set_facecolor('#f0f0f0')
        
        # Calculate the approximate center and radius for the head silhouette
        # based on channel positions
        center_x = np.mean(ch_positions[:, 0])
        center_y = np.mean(ch_positions[:, 1])

        # Calculate INITIAL head radius before adjustment
        initial_distances = np.sqrt((ch_positions[:, 0] - center_x)**2 + 
                                  (ch_positions[:, 1] - center_y)**2)
        head_radius = np.max(initial_distances) * 1.2  # 20% larger than furthest channel
        
        # Parameters for sensor positioning
        vertical_offset = 1.8  # Base vertical offset
        horizontal_spread = 2  # Horizontal spread factor
        vertical_spread = 1.4  # NEW: Factor for vertical spreading
        
        y_relative = ch_positions[:, 1] - center_y

        # Apply transformations to channel positions
        for i in range(len(ch_positions)):
            # Calculate normalized position relative to center
            rel_y = ch_positions[i, 1] - center_y
            
            # Move up by offset proportional to radius
            ch_positions[i, 1] = ch_positions[i, 1] + vertical_offset * head_radius
            
            # Add additional vertical spreading based on relative position
            ch_positions[i, 1] += y_relative[i] * vertical_spread
            
            # Apply horizontal spread from center
            ch_positions[i, 0] = center_x + (ch_positions[i, 0] - center_x) * horizontal_spread
        
        # Recalculate head radius after position adjustment
        distances = np.sqrt((ch_positions[:, 0] - center_x)**2 + 
                          (ch_positions[:, 1] - center_y)**2)
        head_radius = np.max(distances) * 1.2  # 20% larger than furthest channel
        
        # Draw head silhouette (circle)
        head_circle = patches.Circle((center_x, center_y), head_radius, 
                                    fill=False, color='black', linewidth=1.5,
                                    zorder=1)
        ax.add_patch(head_circle)
        
        # Draw simple facial features for orientation
        # Nose (small triangle at the top)
        nose_height = 0.15 * head_radius
        ax.plot([center_x, center_x], 
                [center_y + head_radius, center_y + head_radius + nose_height], 
                'k-', linewidth=1.5, zorder=1)
        
        # Smaller ears with correct orientation
        ear_width = 0.15 * head_radius  # Small width
        ear_height = 0.3 * head_radius  # Small height
        
        # Left ear - "(" shape (opening towards right)
        left_ear = patches.Arc((center_x - head_radius, center_y), 
                              ear_width*2, ear_height*2, 
                              theta1=90, theta2=270, 
                              color='black', linewidth=1.5, zorder=1)
        
        # Right ear - ")" shape (opening towards left)
        right_ear = patches.Arc((center_x + head_radius, center_y), 
                               ear_width*2, ear_height*2, 
                               theta1=270, theta2=90, 
                               color='black', linewidth=1.5, zorder=1)
        
        ax.add_patch(left_ear)
        ax.add_patch(right_ear)
        
        # Extract unique source-detector pairs and create a mapping
        channel_pairs = {}
        for idx, ch_name in enumerate(raw_data.ch_names):
            # Extract just the S*_D* part from the channel name
            parts = ch_name.split(' ')
            if len(parts) >= 1:
                base_name = parts[0]  # This should be the S*_D* part
                channel_pairs[idx] = base_name
            else:
                channel_pairs[idx] = ch_name  # Fallback to full name if parsing fails
        
        # Draw all base channels with simplified labels
        for idx, pos in enumerate(ch_positions):
            ax.scatter(pos[0], pos[1], s=50, color='gray', alpha=0.5, marker='o', zorder=10)
            
            # Use the simplified channel name (just S*_D* part)
            simplified_name = channel_pairs[idx]
            ax.annotate(simplified_name, 
                        (pos[0], pos[1]),
                        textcoords="offset points",
                        xytext=(0,5),
                        ha='center',
                        fontsize=8,
                        zorder=20)
        
        # Draw activations
        if activations:
            cmap = plt.get_cmap('viridis')
            norm = Normalize(vmin=0, vmax=1)
            
            for ch_name, val in activations.items():
                if ch_name in raw_data.ch_names:
                    idx = raw_data.ch_names.index(ch_name)
                    x, y = ch_positions[idx]
                    
                    # Size proportional to area (better scaling)
                    size = 50 + (val ** 0.5) * 1000  # Non-linear adjustment for better visualization
                    
                    ax.scatter(x, y, 
                             s=size,
                             color=cmap(val),
                             alpha=0.7,
                             edgecolor='black',
                             linewidth=1,
                             zorder=15)  # Between base channels and labels
            
            # Color bar
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
            cbar.set_label('Activation Level', fontsize=10)
        
        # Adjust limits automatically to cover the entire head
        padding = 0.2 * head_radius  # More padding to include ears and nose
        ax.set_xlim(center_x - head_radius - padding, center_x + head_radius + padding)
        ax.set_ylim(center_y - head_radius - padding, center_y + head_radius + nose_height + padding)
        ax.set_aspect('equal')
        plt.axis('off')
        plt.tight_layout()
        
        # Convert to base64
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        traceback.print_exc()
        return None
    
def calculate_activations(raw_data):
    """
    Calculate normalized activations (0-1) for each channel
    
    Parameters:
    ----------
    raw_data : mne.io.Raw
        Raw NIRS data
        
    Returns:
    -------
    dict
        Dictionary {channel_name: normalized_activation}
    """
    import numpy as np
    
    # Extract data and calculate absolute mean per channel
    data, _ = raw_data[:, :]
    mean_abs = np.mean(np.abs(data), axis=1)
    
    # Normalize to 0-1 range
    min_val = np.min(mean_abs)
    max_val = np.max(mean_abs)
    normalized = (mean_abs - min_val) / (max_val - min_val + 1e-8)  # +1e-8 avoids division by 0
    
    # Create dictionary
    return {ch: float(normalized[i]) for i, ch in enumerate(raw_data.ch_names)}
def process_nirs_file_with_temporal_validation(file_path, selected_activities):
    """Process NIRS file with temporal validation against bias"""
    # Use the same preprocessing as in analyze_nirs_file
    raw_data = load_nirs_data(file_path)
    
    # Extract features just like in analyze_nirs_file
    features_result = extract_features_from_events(raw_data, ...)
    
    # Instead of apply_machine_learning, use the validation function
    from .nirs_ml import validate_against_temporal_bias
    
    temporal_validation = validate_against_temporal_bias(
        features_result['X_features'], 
        features_result['labels'],
        features_result['feature_names']
    )
    
    # Return both the regular results and validation results
    combined_results = {
        # All the visualizations from regular results
        **temporal_validation['regular_results'],
        # Add temporal validation specific results
        'temporal_validation': {
            'shuffle_mean_accuracy': temporal_validation['shuffle_mean_accuracy'],
            'p_value': temporal_validation['p_value'],
            'significant': temporal_validation['significant']
        }
    }
    
    return combined_results