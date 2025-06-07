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

def create_connectivity_plot(raw_data, feature_result):
    """Create connectivity matrix visualization between channels"""
    try:
        print("=== DEBUGGING: Creating connectivity plot ===")
        
        # Verificar seaborn
        try:
            import seaborn as sns
            print("✓ Seaborn imported successfully")
        except ImportError as e:
            print(f"✗ Error importing seaborn: {e}")
            return None
        
        # Extract channel data
        data_chunks = feature_result.get('data_chunks', [])
        print(f"Data chunks available: {len(data_chunks)}")
        
        if not data_chunks:
            print("✗ No data chunks available for connectivity plot")
            return None
            
        # Calculate correlation matrix between channels
        all_data = np.concatenate(data_chunks, axis=1)  # Concatenate across time
        print(f"All data shape after concatenation: {all_data.shape}")
        
        n_channels = min(20, all_data.shape[0])  # Limit to first 20 channels
        selected_data = all_data[:n_channels, :]
        print(f"Selected data shape: {selected_data.shape}")
        
        # Verificar que hay suficientes datos
        if selected_data.shape[1] < 2:
            print("✗ Insufficient time points for correlation")
            return None
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(selected_data)
        print(f"Correlation matrix shape: {corr_matrix.shape}")
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f',
                   xticklabels=raw_data.ch_names[:n_channels],
                   yticklabels=raw_data.ch_names[:n_channels],
                   cbar_kws={"shrink": .8})
        
        ax.set_title('Channel Connectivity Matrix (Pearson Correlation)', 
                    fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        result = encode_figure_to_base64(fig)
        print(f"✓ Connectivity plot created successfully: {len(result) if result else 0} characters")
        return result
        
    except Exception as e:
        print(f"✗ Error creating connectivity plot: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def create_spectral_analysis_plot(raw_data, event_activations):
    """Create spectral analysis visualization"""
    try:
        print("=== DEBUGGING: Creating spectral analysis plot ===")
        
        # Verificar scipy
        try:
            from scipy import signal
            print("✓ Scipy imported successfully")
        except ImportError as e:
            print(f"✗ Error importing scipy: {e}")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Select representative channels
        n_channels = min(4, len(raw_data.ch_names))
        selected_channels = raw_data.ch_names[:n_channels]
        print(f"Selected channels for spectral analysis: {selected_channels}")
        
        for i, ch_name in enumerate(selected_channels):
            ax = axes[i//2, i%2]
            
            # Get channel data
            ch_idx = raw_data.ch_names.index(ch_name)
            data = raw_data.get_data()[ch_idx, :]
            print(f"Channel {ch_name} data shape: {data.shape}, sampling freq: {raw_data.info['sfreq']}")
            
            # Verificar que hay suficientes datos
            if len(data) < 256:
                print(f"Warning: Channel {ch_name} has insufficient data for spectral analysis")
                continue
            
            # Calculate power spectral density
            freqs, psd = signal.welch(data, raw_data.info['sfreq'], 
                                    nperseg=min(256, len(data)//4))
            
            print(f"Frequency range: {freqs[0]:.4f} - {freqs[-1]:.4f} Hz")
            
            # Plot only relevant frequency range for NIRS (0.01-0.5 Hz)
            freq_mask = (freqs >= 0.01) & (freqs <= 0.5)
            
            if not np.any(freq_mask):
                print(f"Warning: No frequencies in NIRS range for channel {ch_name}")
                continue
            
            ax.loglog(freqs[freq_mask], psd[freq_mask], 'b-', linewidth=2)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power Spectral Density')
            ax.set_title(f'Channel {ch_name}')
            ax.grid(True, alpha=0.3)
            
            # Add frequency band annotations
            ax.axvspan(0.01, 0.08, alpha=0.2, color='red', label='Systemic')
            ax.axvspan(0.08, 0.15, alpha=0.2, color='green', label='Respiratory')
            ax.axvspan(0.15, 0.4, alpha=0.2, color='blue', label='Cardiac')
            
            if i == 0:
                ax.legend()
        
        plt.suptitle('Spectral Analysis by Channel', fontsize=16)
        plt.tight_layout()
        
        result = encode_figure_to_base64(fig)
        print(f"✓ Spectral analysis plot created successfully: {len(result) if result else 0} characters")
        return result
        
    except Exception as e:
        print(f"✗ Error creating spectral analysis plot: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def create_signal_quality_plot(raw_data):
    """Create signal quality assessment visualization"""
    try:
        print("=== DEBUGGING: Creating signal quality plot ===")
        
        # Verificar scipy
        try:
            from scipy import signal as sp_signal
            print("✓ Scipy signal imported successfully")
        except ImportError as e:
            print(f"✗ Error importing scipy signal: {e}")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Get all channel data
        data = raw_data.get_data()
        print(f"Signal quality analysis: {data.shape[0]} channels, {data.shape[1]} time points")
        
        # Verificar que hay suficientes datos
        if data.shape[1] < 100:
            print("✗ Insufficient time points for signal quality analysis")
            return None
        
        # 1. Signal-to-Noise Ratio per channel
        ax1 = axes[0, 0]
        snr_values = []
        
        for i, ch_name in enumerate(raw_data.ch_names):
            try:
                signal_power = np.var(data[i, :])
                
                # Verificar frecuencia de muestreo válida
                if raw_data.info['sfreq'] <= 1.0:
                    print(f"Warning: Very low sampling frequency: {raw_data.info['sfreq']} Hz")
                    # Use a simpler noise estimation
                    noise_power = np.var(np.diff(data[i, :]))
                else:
                    # Estimate noise as high-frequency component
                    try:
                        nyquist = raw_data.info['sfreq'] / 2
                        high_freq = min(0.5, nyquist * 0.8)  # Use 80% of Nyquist or 0.5 Hz
                        b, a = sp_signal.butter(4, high_freq / nyquist, 'high')
                        noise = sp_signal.filtfilt(b, a, data[i, :])
                        noise_power = np.var(noise)
                    except Exception as filter_e:
                        print(f"Filter error for channel {ch_name}: {filter_e}")
                        noise_power = np.var(np.diff(data[i, :]))
                
                snr = 10 * np.log10(signal_power / (noise_power + 1e-12))
                snr_values.append(snr)
                
            except Exception as ch_e:
                print(f"Error processing channel {ch_name}: {ch_e}")
                snr_values.append(0)
        
        print(f"SNR values calculated: min={min(snr_values):.2f}, max={max(snr_values):.2f}")
        
        ax1.bar(range(len(snr_values)), snr_values)
        ax1.set_xlabel('Channel Index')
        ax1.set_ylabel('SNR (dB)')
        ax1.set_title('Signal-to-Noise Ratio by Channel')
        ax1.axhline(y=10, color='r', linestyle='--', label='Good Quality Threshold')
        ax1.legend()
        
        # 2. Signal variance distribution
        ax2 = axes[0, 1]
        variances = np.var(data, axis=1)
        ax2.hist(variances, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Signal Variance')
        ax2.set_ylabel('Number of Channels')
        ax2.set_title('Signal Variance Distribution')
        ax2.axvline(np.mean(variances), color='r', linestyle='--', label='Mean')
        ax2.legend()
        
        # 3. Correlation with mean signal (artifact detection)
        ax3 = axes[1, 0]
        mean_signal = np.mean(data, axis=0)
        correlations = []
        
        for i in range(data.shape[0]):
            try:
                corr = np.corrcoef(data[i, :], mean_signal)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                correlations.append(corr)
            except Exception as corr_e:
                print(f"Correlation error for channel {i}: {corr_e}")
                correlations.append(0.0)
        
        ax3.bar(range(len(correlations)), correlations)
        ax3.set_xlabel('Channel Index')
        ax3.set_ylabel('Correlation with Mean')
        ax3.set_title('Correlation with Mean Signal (Artifact Detection)')
        ax3.axhline(y=0.8, color='r', linestyle='--', label='High Artifact Threshold')
        ax3.legend()
        
        # 4. Signal amplitude distribution
        ax4 = axes[1, 1]
        amplitudes = np.std(data, axis=1)
        ax4.bar(range(len(amplitudes)), amplitudes)
        ax4.set_xlabel('Channel Index')
        ax4.set_ylabel('Signal Amplitude (std)')
        ax4.set_title('Signal Amplitude by Channel')
        
        plt.suptitle('Signal Quality Assessment', fontsize=16)
        plt.tight_layout()
        
        result = encode_figure_to_base64(fig)
        print(f"✓ Signal quality plot created successfully: {len(result) if result else 0} characters")
        return result
        
    except Exception as e:
        print(f"✗ Error creating signal quality plot: {e}")
        import traceback
        traceback.print_exc()
        return None
    
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
        print_available_channels(raw_data)  # Add this line
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

# ...existing code...
def generate_events_plot(raw_data, valid_events, event_info, event_ids, sfreq, max_time):
    """Generate detailed visualization of events covering the full duration."""
    # --- MODIFICATION START: Increase width factor significantly ---
    # Increase figure width substantially to make the initial view cover a smaller time window (e.g., ~120s)
    fig_width = max(15, 0.1 * max_time) # Increased factor from 0.01 to 0.1
    # --- MODIFICATION END ---
    fig_height = 7 # Keep height reasonable (or adjust if needed)
    fig_events = plt.figure(figsize=(fig_width, fig_height))

    # First plot: Raw data visualization with event markers (Full Duration)
    ax1 = plt.subplot(2, 1, 1)

    # Determine optimal plot parameters
    n_channels_to_show = min(5, len(raw_data.ch_names))
    ch_names_to_show = raw_data.ch_names[:n_channels_to_show]

    plot_start_time = 0.0
    plot_duration = max_time

    # Extract data for plotting (Full Duration)
    start_sample = int(plot_start_time * sfreq)
    end_sample = int((plot_start_time + plot_duration) * sfreq)
    end_sample = min(end_sample, raw_data.n_times) 
    
    if start_sample < raw_data.n_times:
        data, times = raw_data[:n_channels_to_show, start_sample:end_sample]
        times = times + plot_start_time 
    else:
        data = np.array([[] for _ in range(n_channels_to_show)])
        times = np.array([])

    # Plot the data lines
    for i, ch_data in enumerate(data):
        if ch_data.size > 0: 
            ch_data_mean = np.mean(ch_data)
            ch_data_std = np.std(ch_data)
            ch_data_norm = (ch_data - ch_data_mean) / (ch_data_std if ch_data_std > 0 else 1)
            ax1.plot(times, ch_data_norm + i*3, linewidth=0.5)

    ax1.set_yticks(np.arange(0, n_channels_to_show*3, 3))
    ax1.set_yticklabels(ch_names_to_show)

    # --- MODIFICATION START: Adjust max_labels based on new width ---
    # Allow more labels for a wider plot, proportional to width
    max_labels = int(fig_width * 1.5) # Adjust factor (e.g., 1.5 labels per inch) as needed
    # --- MODIFICATION END ---
    visible_events = valid_events 

    if len(visible_events) > max_labels:
        step = max(1, len(visible_events) // max_labels)
        events_to_label = visible_events[::step]
        for event in visible_events:
             event_time = event[0] / sfreq
             ax1.axvline(event_time, color='r', linestyle='--', alpha=0.4, linewidth=0.5) 
    else:
        events_to_label = visible_events
        for event in visible_events:
             event_time = event[0] / sfreq
             ax1.axvline(event_time, color='r', linestyle='--', alpha=0.4, linewidth=0.5) 

    # Add labels for selected events
    for event in events_to_label:
        event_time = event[0] / sfreq
        event_code = event[2]
        event_desc = next((k for k, v in event_ids.items() if v == event_code), f"Code {event_code}")
        
        mins = int(event_time) // 60
        secs = event_time % 60
        time_str = f"{mins:02d}:{secs:04.1f}"

        ax1.text(event_time, n_channels_to_show*3, f"{event_desc}\n({time_str})",
                rotation=90, verticalalignment='bottom', fontsize=10) 

    ax1.set_xlabel('Time (s)')
    ax1.set_title(f'Raw Data with Event Markers ({len(visible_events)} events total)')
    ax1.set_xlim(plot_start_time, plot_start_time + plot_duration)
    
    # Set x-axis ticks at 60-second intervals
    tick_interval = 60
    max_tick = int(plot_duration // tick_interval) * tick_interval
    ticks = np.arange(plot_start_time, max_tick + tick_interval, tick_interval)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([str(int(t)) for t in ticks]) 
    
    # Remove the "Showing X.Xs - Y.Ys..." text as it's always full duration now
    # if plot_start_time > 0 or plot_start_time + plot_duration < max_time:
    #     ax1.text(...) # This block is removed/commented out

    # Second plot: timeline showing events with labels (Full Duration)
    ax2 = plt.subplot(2, 1, 2)
    
    event_types = sorted(list(set([e['description'] for e in event_info])))
    colors = plt.cm.tab10(np.linspace(0, 1, len(event_types)))
    color_map = {event_type: colors[i] for i, event_type in enumerate(event_types)}
    
    timeline_start = plot_start_time
    timeline_end = plot_start_time + plot_duration
    
    visible_events_timeline = event_info # All events are relevant now

    # --- MODIFICATION START: Adjust max_event_labels for timeline based on new width ---
    # Allow more labels for a wider plot, proportional to width
    max_event_labels = int(fig_width * 0.5) # Adjust density factor (e.g., 0.5 labels per inch)
    # --- MODIFICATION END ---
    
    if len(visible_events_timeline) > max_event_labels:
        step = max(1, len(visible_events_timeline) // max_event_labels)
        events_to_label = visible_events_timeline[::step]
    else:
        events_to_label = visible_events_timeline
    
    # Plot events as colored spans
    for event in event_info:
        desc = event['description']
        onset = event['onset']
        duration = event['duration'] if event['duration'] > 0 else 5.0 # Default duration if 0
        ax2.axvspan(onset, onset + duration, alpha=0.3, color=color_map[desc])
    
    # Add labels for selected events
    for event in events_to_label:
        desc = event['description']
        onset = event['onset'] 
        duration = event['duration'] if event['duration'] > 0 else 5.0
        
        label_x = onset + duration / 2
        
        # Determine rotation based on duration relative to total time span
        relative_duration = duration / max_time
        rotate_label = relative_duration < 0.01 # Rotate if event is less than 1% of total time (adjust threshold if needed)

        ax2.text(label_x, 0.5, desc, 
                 horizontalalignment='center', 
                 verticalalignment='center',
                 rotation=90 if rotate_label else 0,
                 fontsize=10, color='black', # Smaller font
                 transform=ax2.get_xaxis_transform())
    
    ax2.set_xlim(timeline_start, timeline_end) 
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time (s)')
    ax2.set_yticks([])
    ax2.set_title(f'Event Timeline ({len(visible_events_timeline)} events total)')
    ax2.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Set x-axis ticks at 60-second intervals for ax2
    ax2.set_xticks(ticks) 
    ax2.set_xticklabels([str(int(t)) for t in ticks]) 
    
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[t], alpha=0.3) for t in event_types]
    ax2.legend(handles, event_types, loc='upper right', fontsize=11)
    
    # Remove the "Showing X.Xs - Y.Ys..." text
    # if timeline_start > 0 or timeline_end < max_time:
    #     ax2.text(...) # This block is removed/commented out
    
    plt.tight_layout(pad=1.5)
    return fig_events

def extract_features_from_events(raw_data, valid_events, event_ids, tmin=-5.0, tmax=20.0):
    """
    Extract features from events in raw data using all available channels directly.
    Also calculates average channel activations per event type.
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

    # For visualization by condition and activation calculation
    condition_data = {cond: [] for cond in event_ids}

    # Extract data for each event
    print(f"Processing {len(valid_events)} valid events...")
    for i, event in enumerate(valid_events):
        onset_sample = event[0]
        event_code = event[2]
        # Find condition name from event_ids using the event_code
        condition = next((k for k, v in event_ids.items() if v == event_code), f"Unknown_{event_code}")

        # Calculate start and end samples for this event
        start = onset_sample + start_offset
        end = onset_sample + end_offset

        # Extract data for this time window
        # Ensure start and end are within bounds
        start = max(0, start)
        end = min(raw_data.n_times, end)
        if start >= end:
            print(f"  Skipping event {i+1} ({condition}): Invalid time window [{start/sfreq:.1f}s - {end/sfreq:.1f}s].")
            continue

        data = raw_data.get_data(start=start, stop=end)

        # Store for later
        data_chunks.append(data)
        labels.append(event_code)
        condition_names.append(condition)
        if condition in condition_data:
            condition_data[condition].append(data) # Store raw data segment for this event

    # Check if any data was actually extracted
    if not data_chunks:
        print("Error: No data chunks could be extracted for any event.")
        # Return an empty structure or raise an error
        return {
            'X_features': np.array([]),
            'feature_names': [],
            'labels': np.array([]),
            'condition_names': [],
            'time_points': np.array([]),
            'plots': {'average_response': None, 'channels': None},
            'region_data': {},
            'event_activations': {},
            'n_events': 0,
            'event_ids': event_ids,
            'error': 'No data extracted for events'
        }


    # Generate average response by condition plot
    fig_avg, ax = plt.subplots(figsize=(12, 7))
    time_points = np.linspace(tmin, tmax, end_offset - start_offset)

    for condition, condition_chunks in condition_data.items():
        if condition_chunks:
            # Average across events and channels for this condition
            try:
                avg_data = np.mean(np.array(condition_chunks), axis=0).mean(axis=0)
                if avg_data.shape == time_points.shape:
                    ax.plot(time_points, avg_data, linewidth=2, label=condition)
                else:
                    print(f"  Warning: Shape mismatch for average response plot for condition '{condition}'. Expected {time_points.shape}, got {avg_data.shape}.")
            except Exception as e:
                print(f"  Error calculating average response for condition '{condition}': {e}")


    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Average Response by Condition')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.tight_layout()
    avg_response_plot = encode_figure_to_base64(fig_avg)

    # --- Calculate Per-Event Activations ---
    event_activations = {}
    print("Calculating activations per event type...")
    for condition, condition_chunks in condition_data.items():
        if condition_chunks:
            try:
                # Stack all data segments for this condition: (n_events, n_channels, n_times)
                condition_array = np.array(condition_chunks)
                # Calculate mean absolute signal across time and events for each channel
                # Resulting shape: (n_channels,)
                mean_abs_signal_per_channel = np.mean(np.abs(condition_array), axis=(0, 2))

                # Normalize activations (0-1) for this specific event type
                min_val = np.min(mean_abs_signal_per_channel)
                max_val = np.max(mean_abs_signal_per_channel)
                # Add epsilon to prevent division by zero if all activations are the same
                normalized_activations = (mean_abs_signal_per_channel - min_val) / (max_val - min_val + 1e-9)

                # Create dictionary {channel_name: activation} for this event
                activations_dict = {
                    ch_name: float(normalized_activations[i])
                    for i, ch_name in enumerate(raw_data.ch_names)
                    if i < len(normalized_activations) # Safety check
                }
                event_activations[condition] = activations_dict
                print(f"  Calculated activations for event: {condition}")
            except Exception as e:
                 print(f"  Error calculating activations for event '{condition}': {e}")
                 event_activations[condition] = None
        else:
            print(f"  Skipping activation calculation for event '{condition}': No data.")
            event_activations[condition] = None # Or an empty dict {}
    # --- End Per-Event Activation Calculation ---

    # Extract unique channel identifiers
    unique_channels = []
    for ch_name in raw_data.ch_names:
        parts = ch_name.split(' ')
        if len(parts) >= 1:
            channel = parts[0]  # Extract only the S*_D* identifier
            if channel not in unique_channels:
                unique_channels.append(channel)

    # Sort channels for better visualization
    unique_channels.sort()

    # Generate channel analysis plot
    num_channels_to_plot = min(10, len(unique_channels))
    fig_channels, axs = plt.subplots(num_channels_to_plot, 1, figsize=(15, 4 * num_channels_to_plot), squeeze=False) # Ensure axs is always 2D array
    axs = axs.flatten() # Flatten to 1D array for easy iteration

    region_data = {}

    # Only show the first few channels to avoid cluttering the graph
    channels_to_show = unique_channels[:num_channels_to_plot]

    for i, channel in enumerate(channels_to_show):
        region_data[channel] = {}
        ax = axs[i]

        # Identify indexes for this channel (S*_D*)
        channel_picks = [idx for idx, ch in enumerate(raw_data.ch_names) if ch.startswith(channel + ' ')]

        if channel_picks:
            for condition, condition_chunks in condition_data.items():
                if condition_chunks:
                    try:
                        # Extract channel data for all events in this condition
                        channel_condition_data = [chunk[channel_picks, :] for chunk in condition_chunks]

                        if channel_condition_data:
                            # Average across events, then across selected channels (e.g., 760nm and 850nm for the same S-D pair)
                            channel_avg = np.mean([np.mean(chunk, axis=0) for chunk in channel_condition_data], axis=0)
                            if channel_avg.shape == time_points.shape:
                                ax.plot(time_points, channel_avg, linewidth=2, label=condition)

                                # Store for results (average over time)
                                region_data[channel][condition] = {
                                    'mean': float(np.mean(channel_avg)),
                                    'peak': float(np.max(channel_avg)),
                                    'std': float(np.std(channel_avg))
                                }
                            else:
                                print(f"  Warning: Shape mismatch for channel plot for condition '{condition}', channel '{channel}'. Expected {time_points.shape}, got {channel_avg.shape}.")
                    except Exception as e:
                        print(f"  Error processing channel plot for condition '{condition}', channel '{channel}': {e}")


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

        # For each channel (S*_D* pair)
        for channel in unique_channels:
            # For each wavelength associated with this channel
            channel_wavelength_picks = {}
            for wl in unique_wavelengths:
                 # Find the specific index for this channel and wavelength
                 pick = [idx for idx, ch in enumerate(raw_data.ch_names) if ch == f"{channel} {wl}"]
                 if pick:
                     channel_wavelength_picks[wl] = pick[0] # Store the single index

            # If no specific wavelengths found, use all indices starting with the channel name
            if not channel_wavelength_picks and not unique_wavelengths:
                 all_picks = [idx for idx, ch in enumerate(raw_data.ch_names) if ch.startswith(channel + ' ')]
                 if all_picks:
                     channel_wavelength_picks['all'] = all_picks # Store list of indices

            # Calculate features per wavelength (or 'all')
            for wave_label, picks in channel_wavelength_picks.items():
                # Feature name prefix
                feature_prefix = f"{channel}_{wave_label}"

                # Extract data for these channels/wavelengths
                # If picks is a list (for 'all'), average across them first
                if isinstance(picks, list):
                    data = np.mean(data_chunk[picks, :], axis=0, keepdims=True) # Shape (1, n_times)
                else: # picks is a single index
                    data = data_chunk[[picks], :] # Shape (1, n_times)

                try:
                    # Calculate features (axis=1 is time, axis=0 is channel - only 1 here)
                    baseline_mean = data[:, baseline_indices].mean() if np.any(baseline_indices) and data.shape[1] > 0 else 0

                    # Early window mean
                    early_mean = data[:, early_indices].mean() if np.any(early_indices) and data.shape[1] > 0 else 0
                    features.append(early_mean)
                    if i == 0: feature_names.append(f"{feature_prefix}_early_mean")

                    # Middle window mean
                    middle_mean = data[:, middle_indices].mean() if np.any(middle_indices) and data.shape[1] > 0 else 0
                    features.append(middle_mean)
                    if i == 0: feature_names.append(f"{feature_prefix}_middle_mean")

                    # Late window mean
                    late_mean = data[:, late_indices].mean() if np.any(late_indices) and data.shape[1] > 0 else 0
                    features.append(late_mean)
                    if i == 0: feature_names.append(f"{feature_prefix}_late_mean")

                    # Early slope
                    slope_early = (middle_mean - early_mean) / 5.0 if middle_mean != early_mean else 0
                    features.append(slope_early)
                    if i == 0: feature_names.append(f"{feature_prefix}_slope_early")

                    # Late slope
                    slope_late = (late_mean - middle_mean) / 5.0 if late_mean != middle_mean else 0
                    features.append(slope_late)
                    if i == 0: feature_names.append(f"{feature_prefix}_slope_late")

                    # Amplitude relative to baseline
                    peak_mean = max(early_mean, middle_mean, late_mean)
                    amplitude = peak_mean - baseline_mean
                    features.append(amplitude)
                    if i == 0: feature_names.append(f"{feature_prefix}_amplitude")

                    # Overall variability
                    total_std = np.std(data) if data.size > 0 else 0
                    features.append(total_std)
                    if i == 0: feature_names.append(f"{feature_prefix}_std")

                except Exception as e:
                    print(f"Error calculating feature for {feature_prefix} in event {i+1}: {e}")
                    # Append NaNs or zeros for all features of this channel/wavelength
                    num_expected_features = 7 # early_mean, middle_mean, late_mean, slope_early, slope_late, amplitude, std
                    features.extend([np.nan] * num_expected_features)
                    if i == 0: # Add names only once
                         feature_names.extend([
                             f"{feature_prefix}_early_mean", f"{feature_prefix}_middle_mean",
                             f"{feature_prefix}_late_mean", f"{feature_prefix}_slope_early",
                             f"{feature_prefix}_slope_late", f"{feature_prefix}_amplitude",
                             f"{feature_prefix}_std"
                         ])


        # Handle NaN values for the entire event's feature vector
        features = np.array(features)
        if np.any(np.isnan(features)):
            # print(f"Warning: NaN found in features for event {i+1}. Replacing with 0.")
            features = np.nan_to_num(features, nan=0.0)

        X_features.append(features)

    # Convert to numpy arrays
    X_features = np.array(X_features)
    labels = np.array(labels)

    # Final check on feature matrix shape consistency
    if len(X_features) > 0 and X_features.shape[1] != len(feature_names):
         print(f"CRITICAL WARNING: Feature matrix columns ({X_features.shape[1]}) do not match feature names count ({len(feature_names)}). Check feature extraction logic.")
         # Decide how to handle: return error, try to fix, etc.
         # For now, let's truncate feature_names if it's longer, or pad X_features if shorter (less safe)
         if X_features.shape[1] < len(feature_names):
             print("  Truncating feature names list.")
             feature_names = feature_names[:X_features.shape[1]]
    print("=== DEBUGGING: Creating advanced plots ===")
    
    # Crear plots avanzados con debugging
    connectivity_plot = create_connectivity_plot(raw_data, {'data_chunks': data_chunks})
    print(f"Connectivity plot result: {'SUCCESS' if connectivity_plot else 'FAILED'}")
    
    spectral_plot = create_spectral_analysis_plot(raw_data, event_activations)
    print(f"Spectral plot result: {'SUCCESS' if spectral_plot else 'FAILED'}")
    
    quality_plot = create_signal_quality_plot(raw_data)
    print(f"Quality plot result: {'SUCCESS' if quality_plot else 'FAILED'}")

    features_result = {
        'X_features': X_features,
        'feature_names': feature_names,
        'labels': labels,
        'condition_names': condition_names,
        'time_points': time_points,
        'data_chunks': data_chunks,  
        'plots': {
            'average_response': avg_response_plot,
            'channels': channels_plot,
            'connectivity': connectivity_plot,
            'spectral_analysis': spectral_plot,
            'signal_quality': quality_plot
        },
        'region_data': region_data,
        'event_activations': event_activations,
        'n_events': len(valid_events),
        'event_ids': event_ids
    }

    print("=== DEBUGGING: Final plots in result ===")
    for plot_name, plot_data in features_result['plots'].items():
        print(f"{plot_name}: {'AVAILABLE' if plot_data else 'NULL'}")

    return features_result

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
        from .nirs_ml import apply_machine_learning, generate_interpretation_metadata

        raw_data = load_nirs_data(file_path)
        if raw_data is not None:
            # If no specific activities requested, use all non-boundary annotations
            if not activities:
                annotations_set = set([a['description'] for a in raw_data.annotations
                                  if not a['description'].lower().endswith('boundary')])
                activities = sorted(list(annotations_set))
                print(f"No activities specified, using all found annotations: {activities}")

            # Apply annotation mapping if provided
            if annotation_map:
                raw_data = map_numeric_annotations_to_descriptive(raw_data, annotation_map)
                activities = sorted([a['description'] for a in raw_data.annotations if not a['description'].lower().endswith('boundary')])
                print(f"Using activities after potential mapping: {activities}")

            # Create event_ids dictionary from the final activity list
            event_ids = {activity: i+1 for i, activity in enumerate(activities)}
            if not event_ids:
                 return {'error': 'No valid activities found or specified after mapping.'}

            # Extract events and create visualizations
            events_result = extract_events_and_visualize(raw_data, event_ids)
            if 'error' in events_result:
                if events_result['error'] == 'No matching events found':
                     print("No matching events found, cannot proceed with feature extraction or ML.")
                     return {
                         'error': events_result['error'],
                         'message': events_result.get('message'),
                         'available_annotations': events_result.get('available_annotations'),
                         'plots': {'events': events_result.get('events_plot')}
                     }
                else:
                    return events_result

            # Extract features AND per-event activations from events
            features_result = extract_features_from_events(
                raw_data,
                events_result['valid_events'],
                event_ids
            )
            
            # Check for errors during feature extraction
            if features_result.get('error'):
                 return {'error': f"Feature extraction failed: {features_result['error']}"}

            # Get the per-event activations
            event_activations = features_result.get('event_activations')

            # Generate brain visualizations (one per event)
            brain_visualizations_by_event = create_brain_visualization(raw_data, event_activations)

            # Get interpretation metadata
            interpretation_data = generate_interpretation_metadata(
                features_result['feature_names'],
                raw_data,
                None
            )

            # Apply machine learning if we have enough data
            ml_results = None
            n_samples_final = features_result['X_features'].shape[0]
            n_labels_final = len(np.unique(features_result['labels']))

            print(f"=== DEBUGGING ML PREPARATION ===")
            print(f"Samples: {n_samples_final}, Classes: {n_labels_final}")
            print(f"Features shape: {features_result['X_features'].shape}")
            print(f"Labels shape: {features_result['labels'].shape}")

            if n_samples_final > 2 and n_labels_final > 1:
                print(f"Proceeding with ML: {n_samples_final} samples, {n_labels_final} classes.")
                ml_results = apply_machine_learning(
                    features_result['X_features'],
                    features_result['labels'],
                    features_result['feature_names']
                )
                print(f"=== DEBUGGING ML RESULTS ===")
                print(f"ML results keys: {list(ml_results.keys()) if ml_results else 'None'}")
                print(f"ML plots keys: {list(ml_results.get('plots', {}).keys()) if ml_results else 'None'}")
                print(f"Top features: {ml_results.get('top_features', [])[:5] if ml_results else 'None'}")
                print(f"Best classifier: {ml_results.get('best_classifier') if ml_results else 'None'}")
                print(f"Accuracy: {ml_results.get('accuracy') if ml_results else 'None'}")
            else:
                 print(f"Skipping ML: Insufficient data after feature extraction ({n_samples_final} samples, {n_labels_final} classes).")

            # *** CRITICAL FIX: Combine plots correctly ***
            all_plots = {
                'events': events_result['events_plot'],
                'average_response': features_result['plots']['average_response'],
                'brain_visualizations_by_event': brain_visualizations_by_event,
                'channels': features_result['plots']['channels'],
                'connectivity': features_result['plots'].get('connectivity'),
                'spectral_analysis': features_result['plots'].get('spectral_analysis'),
                'signal_quality': features_result['plots'].get('signal_quality')
            }

            # Add ML plots if available
            if ml_results and ml_results.get('plots'):
                all_plots.update(ml_results['plots'])
                print(f"Added ML plots: {list(ml_results['plots'].keys())}")

            # *** CRITICAL FIX: Return the correct structure ***
            results = {
                'plots': all_plots,
                'features': {
                    'shape': features_result['X_features'].shape,
                    'feature_count': len(features_result['feature_names']),
                    'top_features': ml_results.get('top_features', []) if ml_results else []
                },
                'ml_results': ml_results,  # *** ADD THIS LINE ***
                'channel_data': features_result['region_data'],
                'interpretation': interpretation_data,
                'metadata': {
                    'n_channels': len(raw_data.ch_names),
                    'n_events': len(events_result['valid_events']),
                    'sampling_frequency': raw_data.info['sfreq'],
                    'event_counts': events_result['event_stats']['events_by_type']
                }
            }

            print(f"=== DEBUGGING FINAL RESULTS ===")
            print(f"Final results keys: {list(results.keys())}")
            print(f"ML results included: {'ml_results' in results}")
            print(f"Plots keys: {list(results['plots'].keys())}")

            if not ml_results:
                 results['warning'] = 'Insufficient data for machine learning analysis'

            return results
        else:
            return {'error': 'Failed to load NIRS data.'}
    except Exception as e:
        print(f"CRITICAL ERROR in analyze_nirs_file: {str(e)}")
        print(traceback.format_exc())
        return {
            'error': f'Analysis failed due to an unexpected error: {str(e)}'
        }

def create_brain_visualization(raw_data, event_activations): # Removed baseline_event_name parameter
    """
    Generates a 2D visualization showing the DIFFERENCE in NIRS channels activation
    levels between 'Left_hand_lift' and 'Right_hand_lift'.

    Parameters:
    ----------
    raw_data : mne.io.Raw
        Loaded NIRS data
    event_activations : dict
        Dictionary where keys are event names and values are dictionaries
        {channel_name: activation_value} (0-1) for that event. Must contain
        'Left_hand_lift' and 'Right_hand_lift'.

    Returns:
    -------
    dict
        Dictionary containing the difference plot {'Left_vs_Right': base64_image},
        or None if the required events are missing or an error occurs.
    """
    # --- Import necessary libraries ---
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib import patches
    from io import BytesIO
    import numpy as np
    import matplotlib.pyplot as plt
    import base64
    import traceback

    plt.switch_backend('Agg') # Ensure non-interactive backend

    # --- Define specific events for comparison ---
    event1_name = 'Left_hand_lift'
    event2_name = 'Right_hand_lift'

    # --- Check input ---
    if not event_activations:
        print("Skipping brain visualization: No event activations provided.")
        return None

    # --- Check for Required Events ---
    if event1_name not in event_activations or event_activations[event1_name] is None:
        print(f"Warning: Event '{event1_name}' not found or has no data. Cannot generate difference plot.")
        return None
    if event2_name not in event_activations or event_activations[event2_name] is None:
        print(f"Warning: Event '{event2_name}' not found or has no data. Cannot generate difference plot.")
        return None

    activations1 = event_activations[event1_name]
    activations2 = event_activations[event2_name]
    output_plots = {}

    # --- Get original channel positions (once) ---
    # ... (Código para obtener ch_positions_orig, center_x_orig, center_y_orig, head_radius - SIN CAMBIOS) ...
    try:
        layout = mne.channels.find_layout(raw_data.info, ch_type='fnirs')
        pos_dict = {name: pos[:2] for name, pos in zip(layout.names, layout.pos)}
        ch_positions_orig = np.array([pos_dict.get(name, [np.nan, np.nan]) for name in raw_data.ch_names])

        if np.isnan(ch_positions_orig).any():
             print("Warning: Some channel positions not found in standard layout. Using raw 'loc' values.")
             ch_positions_orig = np.array([ch['loc'][:2] for ch in raw_data.info['chs']])
             if np.isnan(ch_positions_orig).any(): return None

        center_x_orig = np.nanmean(ch_positions_orig[:, 0])
        center_y_orig = np.nanmean(ch_positions_orig[:, 1])
        distances_orig = np.sqrt((ch_positions_orig[:, 0] - center_x_orig)**2 + (ch_positions_orig[:, 1] - center_y_orig)**2)
        valid_distances_orig = distances_orig[~np.isnan(distances_orig)]
        head_radius = np.max(valid_distances_orig) * 1.2 if len(valid_distances_orig) > 0 and np.any(valid_distances_orig > 0) else 0.1
    except Exception as e:
        print(f"Error getting channel positions: {e}. Trying raw 'loc' values.")
        try:
            ch_positions_orig = np.array([ch['loc'][:2] for ch in raw_data.info['chs']])
            if np.isnan(ch_positions_orig).any(): return None
            center_x_orig = np.nanmean(ch_positions_orig[:, 0])
            center_y_orig = np.nanmean(ch_positions_orig[:, 1])
            distances_orig = np.sqrt((ch_positions_orig[:, 0] - center_x_orig)**2 + (ch_positions_orig[:, 1] - center_y_orig)**2)
            valid_distances_orig = distances_orig[~np.isnan(distances_orig)]
            head_radius = np.max(valid_distances_orig) * 1.2 if len(valid_distances_orig) > 0 and np.any(valid_distances_orig > 0) else 0.1
        except Exception as e2:
             print(f"Error getting raw channel positions: {e2}")
             return None

    # --- Extract unique source-detector pairs mapping ---
    channel_pairs = {idx: ch.split(' ')[0] if len(ch.split(' ')) >= 1 else ch
                     for idx, ch in enumerate(raw_data.ch_names)}

    # --- 1. Calculate DIFFERENCES (Event1 - Event2) and Global Range ---
    all_diff_vals = []
    aggregated_diffs = {} # Store aggregated differences

    # Aggregate activations per S*_D* pair and calculate difference
    temp_aggregation1 = {}
    temp_aggregation2 = {}

    # Aggregate for event 1
    for ch_name, val in activations1.items():
        if ch_name in raw_data.ch_names:
            idx = raw_data.ch_names.index(ch_name)
            base_name = channel_pairs[idx]
            if base_name not in temp_aggregation1: temp_aggregation1[base_name] = {'sum': 0, 'count': 0}
            if not np.isnan(val):
                temp_aggregation1[base_name]['sum'] += val
                temp_aggregation1[base_name]['count'] += 1

    # Aggregate for event 2
    for ch_name, val in activations2.items():
        if ch_name in raw_data.ch_names:
            idx = raw_data.ch_names.index(ch_name)
            base_name = channel_pairs[idx]
            if base_name not in temp_aggregation2: temp_aggregation2[base_name] = {'sum': 0, 'count': 0}
            if not np.isnan(val):
                temp_aggregation2[base_name]['sum'] += val
                temp_aggregation2[base_name]['count'] += 1

    # Calculate average difference for common base_names
    common_base_names = set(temp_aggregation1.keys()) & set(temp_aggregation2.keys())
    for base_name in common_base_names:
        if temp_aggregation1[base_name]['count'] > 0 and temp_aggregation2[base_name]['count'] > 0:
            avg1 = temp_aggregation1[base_name]['sum'] / temp_aggregation1[base_name]['count']
            avg2 = temp_aggregation2[base_name]['sum'] / temp_aggregation2[base_name]['count']
            diff = avg1 - avg2
            all_diff_vals.append(diff)
            aggregated_diffs[base_name] = {'diff': diff} # Store avg diff

    # Determine symmetric range around 0 for the colormap based ONLY on this comparison
    if not all_diff_vals:
        print("Warning: No common channels with valid data found between the two events. Cannot generate difference plot.")
        return None
    max_abs_diff = max(abs(d) for d in all_diff_vals)
    global_min_diff = -max_abs_diff
    global_max_diff = max_abs_diff
    if global_max_diff == global_min_diff: global_max_diff += 1e-9; global_min_diff -= 1e-9
    print(f"Activation DIFFERENCE range ({event1_name} vs {event2_name}): [{global_min_diff:.3f}, {global_max_diff:.3f}]")

    # --- Define Diverging Normalization (centered at 0) ---
    cmap = plt.get_cmap('coolwarm') # Blue -> White -> Red
    norm = TwoSlopeNorm(vmin=global_min_diff, vcenter=0, vmax=global_max_diff)

    # --- Generate the PLOT ---
    print(f"Generating difference visualization ({event1_name} vs {event2_name})...")
    fig = None
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_title = f'Activation Difference ({event1_name} vs {event2_name})'
        ax.set_title(plot_title, fontsize=14, pad=20)
        ax.set_facecolor('#f0f0f0')

        # --- Repositioning ---
        ch_positions = ch_positions_orig.copy()
        center_x = center_x_orig; center_y = center_y_orig
        current_center_y = np.nanmean(ch_positions[:, 1])
        target_center_y = center_y + 0.6 * head_radius # Keep position factor
        shift_y = target_center_y - current_center_y
        valid_y_mask = ~np.isnan(ch_positions[:, 1])
        ch_positions[valid_y_mask, 1] += shift_y

        # --- Draw head silhouette and features ---
        # ... (Código para dibujar cabeza, nariz, orejas - SIN CAMBIOS) ...
        head_circle = patches.Circle((center_x, center_y), head_radius, fill=False, color='black', linewidth=1.5, zorder=1); ax.add_patch(head_circle)
        nose_height = 0.15 * head_radius; ax.plot([center_x, center_x], [center_y + head_radius, center_y + head_radius + nose_height], 'k-', linewidth=1.5, zorder=1)
        ear_width = 0.15 * head_radius; ear_height = 0.3 * head_radius
        left_ear = patches.Arc((center_x - head_radius, center_y), ear_width*2, ear_height*2, theta1=90, theta2=270, color='black', linewidth=1.5, zorder=1); ax.add_patch(left_ear)
        right_ear = patches.Arc((center_x + head_radius, center_y), ear_width*2, ear_height*2, theta1=270, theta2=90, color='black', linewidth=1.5, zorder=1); ax.add_patch(right_ear)

        # --- Draw all base channels with simplified labels (at SHIFTED positions) ---
        plotted_base_names = set()
        for idx, pos in enumerate(ch_positions):
            if np.isnan(pos).any(): continue
            simplified_name = channel_pairs[idx]
            if simplified_name not in plotted_base_names:
                ax.scatter(pos[0], pos[1], s=50, color='gray', alpha=0.5, marker='o', zorder=10)
                ax.annotate(simplified_name, (pos[0], pos[1]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, zorder=20)
                plotted_base_names.add(simplified_name)

        # --- Draw DIFFERENCES ---
        for base_name, data in aggregated_diffs.items():
             diff_val = data['diff']
             # Find the corresponding SHIFTED position for this base_name
             try:
                 first_match_idx = next(idx for idx, ch in enumerate(raw_data.ch_names) if channel_pairs[idx] == base_name)
                 x, y = ch_positions[first_match_idx]
             except StopIteration: continue
             if np.isnan(x) or np.isnan(y): continue

             # Size based on ABSOLUTE difference
             abs_diff_normalized = abs(diff_val) / max_abs_diff if max_abs_diff > 1e-9 else 0.0
             size_scale_factor = abs_diff_normalized ** 0.6
             size = 30 + size_scale_factor * 850

             # Color based on actual difference using diverging map
             ax.scatter(x, y, s=max(10, size), color=cmap(norm(diff_val)), alpha=0.75,
                      edgecolor='black', linewidth=0.5, zorder=15)

        # --- Color bar ---
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=15)
        cbar.set_label(f'Activation Difference ({event1_name} - {event2_name})', fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        # --- Final plot adjustments ---
        # ... (Código para ajustar límites, aspecto, etc. - SIN CAMBIOS) ...
        all_y_coords = ch_positions[:, 1]; valid_y_coords = all_y_coords[~np.isnan(all_y_coords)]
        min_sensor_y = np.min(valid_y_coords) if len(valid_y_coords) > 0 else center_y - head_radius
        max_sensor_y = np.max(valid_y_coords) if len(valid_y_coords) > 0 else center_y + head_radius
        max_plot_y = center_y + head_radius + nose_height
        padding_y = 0.1 * head_radius; padding_x = 0.1 * head_radius
        ax.set_xlim(center_x - head_radius - padding_x, center_x + head_radius + padding_x)
        ax.set_ylim(min(min_sensor_y - padding_y, center_y - head_radius - padding_y), max(max_plot_y + padding_y, max_sensor_y + padding_y))
        ax.set_aspect('equal', adjustable='box'); plt.axis('off'); plt.tight_layout(pad=0.5)

        # --- Convert to base64 ---
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        # Store the single plot with a specific key
        output_plots['Left_vs_Right'] = base64.b64encode(buf.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"Error generating difference visualization: {str(e)}")
        traceback.print_exc()
        if fig is not None and plt.fignum_exists(fig.number): plt.close(fig)
        return None # Return None on error

    # Return the dictionary containing the single difference plot
    return output_plots

# ... (rest of the file, including analyze_nirs_file etc.) ...
# --- REMEMBER TO REMOVE/COMMENT OUT the old calculate_activations function if it exists ---
# def calculate_activations(raw_data):
#    ...
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