"""
This module contains functions for processing NIRS data.
It includes methods for loading NIRS files, visualizing events and signals,
and extracting basic features from raw data.
"""

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
        if annot['description'] in event_ids:
            onset = annot['onset']
            
            # Validate event position
            if onset < abs(tmin):
                continue
            if (onset + tmax) > max_time:
                print(f"Event at {onset:.1f}s exceeds recording duration")
                continue
            
            # Convert onset time to sample index
            onset_sample = int(onset * sfreq)
            valid_events.append([onset_sample, 0, event_ids[annot['description']]])
            
            # Store event info for later
            event_info.append({
                'onset': onset,
                'description': annot['description'],
                'duration': annot['duration'],
                'code': event_ids[annot['description']]
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
    Extract features from events in raw data.
    
    Parameters:
    ----------
    raw_data : mne.io.Raw
        Raw NIRS data
    valid_events : list
        List of valid events as [sample_idx, 0, event_code]
    event_ids : dict
        Dictionary mapping event descriptions to numeric IDs
    tmin : float
        Start time relative to event onset for data extraction
    tmax : float
        End time relative to event onset for data extraction
        
    Returns:
    -------
    dict
        Dictionary containing extracted features and related visualizations
    """
    # Replace the current BRAIN_REGIONS definition:

    BRAIN_REGIONS = {
        'prefrontal': ['S1_D1', 'S1_D2', 'S2_D1', 'S2_D2'],
        'central_frontal': ['S3_D3', 'S4_D3', 'S4_D4', 'S5_D4'],
        'lateral_frontal': ['S6_D5', 'S7_D6', 'S8_D7']
    }
    
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
    
    # Generate regional analysis plot
    fig_regions, axs = plt.subplots(len(BRAIN_REGIONS), 1, figsize=(15, 4*len(BRAIN_REGIONS)))
    if len(BRAIN_REGIONS) == 1:
        axs = [axs]
    
    region_data = {}
    
    for i, (region_name, channels) in enumerate(BRAIN_REGIONS.items()):
        region_data[region_name] = {}
        ax = axs[i]
        
        # Identify channels for this region
        region_picks = []
        for ch_name in channels:
            region_picks.extend([idx for idx, ch in enumerate(raw_data.ch_names) 
                                if ch_name in ch])
        
        if region_picks:
            for condition, condition_chunks in condition_data.items():
                if condition_chunks:
                    # Extract regional data for all events in this condition
                    region_condition_data = [chunk[region_picks, :] for chunk in condition_chunks]
                    
                    if region_condition_data:
                        # Average across events and channels
                        region_avg = np.mean([np.mean(chunk, axis=0) for chunk in region_condition_data], axis=0)
                        ax.plot(time_points, region_avg, linewidth=2, label=condition)
                        
                        # Store for results
                        region_data[region_name][condition] = {
                            'mean': float(np.mean(region_avg)),
                            'peak': float(np.max(region_avg)),
                            'std': float(np.std(region_avg))
                        }
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{region_name} region')
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    regions_plot = encode_figure_to_base64(fig_regions)
    
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
        
        # For each brain region
        for region_name, channels in BRAIN_REGIONS.items():
            for wave_label, wave_picks in wave_groups.items():
                region_wave_picks = []
                
                for ch in channels:
                    matching_picks = [idx for idx in wave_picks if ch in raw_data.ch_names[idx]]
                    region_wave_picks.extend(matching_picks)
                
                if not region_wave_picks:
                    # If no channels for this combination, add zeros
                    for _ in range(7):
                        features.append(0)
                        if i == 0:
                            feature_names.append(f"{region_name}_{wave_label}_feature{_}")
                    continue
                
                # Feature name prefix
                feature_prefix = f"{region_name}_{wave_label}"
                
                # Extract data for these channels
                data = data_chunk[region_wave_picks, :]
                
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
                    # Add dummy values to maintain consistency
                    for _ in range(7 - (len(features) % 7)):
                        features.append(0)
                        if i == 0:
                            feature_names.append(f"{feature_prefix}_dummy{_}")
        
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
            'regions': regions_plot
        },
        'region_data': region_data,
        'n_events': len(valid_events),
        'event_ids': event_ids
    }

def analyze_nirs_file(file_path, activities, annotation_map=None):
    """
    Analyze a NIRS file and extract relevant features.

    Parameters:
    ----------
    file_path : str
        The path to the NIRS data file.
    activities : list
        A list of activity names to extract from the file.
    annotation_map : dict, optional
        Mapping from numeric codes to descriptive labels
        
    Returns:
    -------
    dict
        A dictionary containing the analysis results
    """
    try:
        # Import ML functions here to avoid circular imports
        from .nirs_ml import apply_machine_learning
        
        raw_data = load_nirs_data(file_path)
        if raw_data is not None:
            # If no specific activities were requested, use all non-boundary annotations
            if not activities:
                annotations = set([a['description'] for a in raw_data.annotations 
                                  if not a['description'].endswith('boundary')])
                activities = list(annotations)
            
            # Apply annotation mapping if provided
            if annotation_map:
                raw_data = map_numeric_annotations_to_descriptive(raw_data, annotation_map)
                
                # Update activities list with mapped names if needed
                mapped_activities = []
                for activity in activities:
                    if activity in annotation_map:
                        mapped_activities.append(annotation_map[activity])
                    else:
                        mapped_activities.append(activity)
                activities = mapped_activities
            
            # Create event_ids dictionary from activities list
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
            BRAIN_REGIONS = {
                'frontal': ['S1_D1', 'S1_D3', 'S2_D1', 'S2_D2', 'S2_D4', 'S3_D2'],
                'temporal': ['S4_D3', 'S4_D5', 'S5_D4', 'S6_D5', 'S6_D7', 'S7_D6'],
                'parietal': ['S8_D7', 'S8_D9', 'S9_D8']
            }

            from .nirs_ml import generate_interpretation_metadata
            interpretation_data = generate_interpretation_metadata(
                features_result['feature_names'],
                raw_data,
                BRAIN_REGIONS
            )

            # Apply machine learning if we have enough data
            if features_result['X_features'].shape[0] > 2 and len(np.unique(features_result['labels'])) > 1:
                ml_results = apply_machine_learning(
                    features_result['X_features'], 
                    features_result['labels'],
                    features_result['feature_names']
                )
                print(f"[DEBUG] Top features received in analyze_nirs_file: {ml_results.get('top_features', [])}")
                # Combine results
                combined_results = {
                    **events_result['event_stats'],
                    'features': {
                        'shape': features_result['X_features'].shape,
                        'feature_count': len(features_result['feature_names']),
                        'top_features': ml_results.get('top_features', [])
                    },
                    'plots': {
                        'events': events_result['events_plot'],
                        'average_response': features_result['plots']['average_response'],
                        'regions': features_result['plots']['regions'],
                        'classifier_comparison': ml_results.get('plots', {}).get('classifier_comparison'),
                        'confusion_matrix': ml_results.get('plots', {}).get('confusion_matrix'),
                        'feature_importance': ml_results.get('plots', {}).get('feature_importance'),
                        'learning_curve': ml_results.get('plots', {}).get('learning_curve')
                    },
                    'region_data': features_result['region_data'],
                    'best_classifier': ml_results.get('best_classifier'),
                    'interpretation': interpretation_data,
                    'accuracy': ml_results.get('accuracy'),
                    'ml_params': ml_results.get('params', {})
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
                        'regions': features_result['plots']['regions'],
                    },
                    'region_data': features_result['region_data'],
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