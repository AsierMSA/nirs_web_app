"""
This module contains functions and classes for analyzing NIRS data.
It includes methods for processing NIRS files, extracting features,
and preparing data for further analysis or visualization.
"""

import traceback
import mne
import numpy as np
import os
import matplotlib.pyplot as plt
import io
import base64
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive backend
# Add this function somewhere near the beginning of the file

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

def direct_extract_features(raw_data, event_ids):
    """
    Extract features directly from raw data without using epochs.
    This approach avoids the epoch dropping issues.
    """
    # Variables to store results
    features = []
    labels = []
    data_chunks = []
    condition_names = []
    
    # Define brain regions of interest by channels
    BRAIN_REGIONS = {
        'frontal': ['S1_D1', 'S1_D3', 'S2_D1', 'S2_D2', 'S2_D4', 'S3_D2'],
        'temporal': ['S4_D3', 'S4_D5', 'S5_D4', 'S6_D5', 'S6_D7', 'S7_D6'],
        'parietal': ['S8_D7', 'S8_D9', 'S9_D8']
    }
    
    # Parameters (same as in analyze_nirs_data)
    tmin = -5.0
    tmax = 20.0
    sfreq = raw_data.info['sfreq']
    
    # Replace the event extraction and visualization section (around lines 78-105) with this enhanced version:

    # Extract valid events
    valid_events = []
    event_info = []
    max_time = raw_data.times[-1]
    
    # Check if there are any annotations at all
    if len(raw_data.annotations) == 0:
        print("WARNING: No annotations found in the file!")
        return {
            'error': 'No events/annotations found in the file',
            'message': 'The NIRS file does not contain any event markers or annotations. Please check that your file has been properly annotated.',
            'total_duration': max_time
        }
    
    # Print all available annotations for debugging
    print(f"All annotations in file ({len(raw_data.annotations)}):")
    for i, annot in enumerate(raw_data.annotations[:20]):  # Show first 20 to avoid clutter
        print(f"  {i+1}. '{annot['description']}' at {annot['onset']:.1f}s (duration: {annot['duration']:.1f}s)")
    if len(raw_data.annotations) > 20:
        print(f"  ... and {len(raw_data.annotations) - 20} more")
    
    # Count annotations per description
    descriptions_count = {}
    for annot in raw_data.annotations:
        desc = annot['description']
        descriptions_count[desc] = descriptions_count.get(desc, 0) + 1
    
    print("\nAnnotation counts by description:")
    for desc, count in descriptions_count.items():
        print(f"  '{desc}': {count}")
    
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
            if onset < abs(tmin):  # Need enough pre-stimulus data
                print(f"Event at {onset:.1f}s has insufficient pre-stimulus data")
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
        # If no valid events found, give a detailed error message
        if found_activities:
            return {
                'error': 'No valid events found within time boundaries',
                'message': f'Found {len(found_activities)} matching activities, but none have sufficient data before/after to extract features.',
                'total_duration': max_time,
                'found_activities': found_activities,
                'missing_activities': missing_activities
            }
        else:
            return {
                'error': 'No matching events found',
                'message': f'None of the requested activities ({", ".join(event_ids.keys())}) were found in the file annotations.',
                'total_duration': max_time,
                'available_annotations': list(descriptions_count.keys())
            }
    
    # Generate detailed diagnostic visualization of events
    # Create a more informative plot showing events with labels
    # Replace the plot events section with this compatibility fix:

    # Generate detailed diagnostic visualization of events
    fig_events = plt.figure(figsize=(15, 8))
    
    # First plot: Create a custom raw data visualization instead of using raw.plot()
    ax1 = plt.subplot(2, 1, 1)
    
    # Get a subset of channels to display (first 5 channels for simplicity)
    n_channels_to_show = min(5, len(raw_data.ch_names))
    ch_names_to_show = raw_data.ch_names[:n_channels_to_show]
    
    # Plot data for a subset of time
    plot_duration = min(60, max_time)  # Show first minute or less
    
    # Get data for plotting
    start_sample = 0
    end_sample = int(plot_duration * sfreq)
    data, times = raw_data[:n_channels_to_show, start_sample:end_sample]
    
    # Plot the data lines
    for i, ch_data in enumerate(data):
        # Normalize for better visualization
        ch_data = ch_data - np.mean(ch_data)
        ch_data = ch_data / (np.std(ch_data) if np.std(ch_data) > 0 else 1)
        # Plot with offset for visibility
        ax1.plot(times, ch_data + i*3, linewidth=0.5)
        
    # Add channel labels
    ax1.set_yticks(np.arange(0, n_channels_to_show*3, 3))
    ax1.set_yticklabels(ch_names_to_show)
    
    # Mark events on the plot
    for event in valid_events:
        event_time = event[0] / sfreq
        if event_time <= plot_duration:
            event_code = event[2]
            # Find event description
            event_desc = next((k for k, v in event_ids.items() if v == event_code), f"Code {event_code}")
            ax1.axvline(event_time, color='r', linestyle='--', alpha=0.7)
            ax1.text(event_time, n_channels_to_show*3, event_desc, 
                    rotation=90, verticalalignment='bottom', fontsize=8)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Raw Data with Event Markers')
    ax1.set_xlim(0, plot_duration)
    
    # Second plot: timeline showing events with labels (this part remains the same)
    ax2 = plt.subplot(2, 1, 2)
    
    # Create color map for different event types
    event_types = sorted(list(set([e['description'] for e in event_info])))
    colors = plt.cm.tab10(np.linspace(0, 1, len(event_types)))
    color_map = {event_type: colors[i] for i, event_type in enumerate(event_types)}
    
    # Plot events as colored spans
    for event in event_info:
        desc = event['description']
        onset = event['onset']
        duration = event['duration'] if event['duration'] > 0 else 5.0  # Default duration if 0
        ax2.axvspan(onset, onset + duration, alpha=0.3, color=color_map[desc])
        ax2.text(onset + duration/2, 0.5, desc, 
                 horizontalalignment='center', verticalalignment='center',
                 rotation=90 if duration < 10 else 0,
                 fontsize=9, color='black', transform=ax2.get_xaxis_transform())
    
    # Customize timeline plot
    ax2.set_xlim(0, min(max_time, 120))  # Show first 2 minutes or less
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time (s)')
    ax2.set_yticks([])
    ax2.set_title('Event Timeline')
    ax2.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Create a legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[t], alpha=0.3) for t in event_types]
    ax2.legend(handles, event_types, loc='upper right')
    
    plt.tight_layout()

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.tight_layout(pad=1.5) 
    events_plot = encode_figure_to_base64(fig_events)
    
    # Add event statistics to return data
    event_stats = {
        'total_events': len(valid_events),
        'events_by_type': {desc: len([e for e in event_info if e['description'] == desc]) for desc in event_types},
        'time_span': [min([e['onset'] for e in event_info]), max([e['onset'] for e in event_info])]
    }
    
    # 1. DIRECT DATA EXTRACTION APPROACH
    print(f"Processing {len(valid_events)} valid events...")
    
    # Calculate start and end samples for the time window
    start_offset = int(tmin * sfreq)  # Convert to samples (negative for pre-stimulus)
    end_offset = int(tmax * sfreq)    # Convert to samples (positive for post-stimulus)
    
    # For visualization by condition
    condition_data = {cond: [] for cond in event_ids}
    
    # For each event, extract data directly
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
    
    # 2. Generate average response by condition plot
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
    
    # 3. Generate regional analysis plot
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
        ax.set_title(f' {region_name} region')
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    regions_plot = encode_figure_to_base64(fig_regions)
    
    # 4. FEATURE EXTRACTION
    # Define time windows for different phases (same as analyze_nirs_data)
    baseline_indices = (time_points >= -5) & (time_points < 0)
    early_indices = (time_points >= 1) & (time_points <= 4)
    middle_indices = (time_points >= 5) & (time_points <= 10)
    late_indices = (time_points >= 11) & (time_points <= 15)
    
    feature_names = []
    X_features = []
    
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
                    # Baseline
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
        
        # Check for NaNs
        features = np.array(features)
        if np.any(np.isnan(features)):
            features = np.nan_to_num(features, nan=0.0)
        
        X_features.append(features)
    
    # Convert to numpy arrays
    X_features = np.array(X_features)
    labels = np.array(labels)
    
    # If we have enough samples, do a simple classification
    if len(X_features) > 2 and len(np.unique(labels)) > 1:
        # Feature selection
        k_features = min(X_features.shape[0]-1, X_features.shape[1])
        selector = SelectKBest(f_classif, k=k_features)
        X_selected = selector.fit_transform(X_features, labels)
        
        # Classification results
        classifiers = {
            'Naive Bayes': GaussianNB(),
            'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
            'Ridge': RidgeClassifier(alpha=1.0),
            'SVM': SVC(kernel='linear', C=0.1)
        }
        
        # Cross-validation if we have enough samples
        results_dict = {}
        y_true = []
        y_pred = {}
        
        if X_selected.shape[0] > 3:  # Only attempt if we have enough samples
            loo = LeaveOneOut()
            
            for name, clf in classifiers.items():
                correct = 0
                y_pred[name] = []
                
                for train_idx, test_idx in loo.split(X_selected):
                    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
                    y_train, y_test = labels[train_idx], labels[test_idx]
                    
                    # Train and predict
                    clf.fit(X_train, y_train)
                    pred = clf.predict(X_test)
                    correct += (pred[0] == y_test[0])
                    
                    if name == list(classifiers.keys())[0]:  # Only add once
                        y_true.append(y_test[0])
                    y_pred[name].append(pred[0])
                
                accuracy = correct / len(labels)
                results_dict[name] = accuracy
            
            # Visualize comparison
            fig_clf = plt.figure(figsize=(10, 6))
            plt.bar(results_dict.keys(), results_dict.values())
            plt.xlabel('Classifier')
            plt.ylabel('Accuracy')
            plt.title('Classifier Comparison')
            plt.ylim(0, 1)
            for i, (k, v) in enumerate(results_dict.items()):
                plt.text(i, v + 0.05, f'{v:.2f}', ha='center')
            plt.tight_layout()
            classifier_plot = encode_figure_to_base64(fig_clf)
            
            # Best classifier results
            best_classifier_name = max(results_dict, key=results_dict.get)
            best_accuracy = results_dict[best_classifier_name]
            
            # Only the relevant section is modified

# Replace the confusion matrix section (around line 356) with this improved version:
            # Confusion matrix
            best_y_pred = y_pred[best_classifier_name]
            
            # Create a mapping from numeric labels to string labels
            label_mapping = {v: k for k, v in event_ids.items()}
            
            # Convert numeric labels to string labels for better visualization
            y_true_labels = [label_mapping.get(y, f"Unknown-{y}") for y in y_true]
            y_pred_labels = [label_mapping.get(y, f"Unknown-{y}") for y in best_y_pred]
            
            # Get unique labels in the correct order
            unique_labels = sorted(list(set(y_true_labels + y_pred_labels)))
            
            try:
                # Create confusion matrix
                cm = confusion_matrix(y_true_labels, y_pred_labels, labels=unique_labels)
                
                # Create a more visually appealing figure
                plt.figure(figsize=(10, 8))
                
                # Use ConfusionMatrixDisplay for better visualization
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
                disp.plot(cmap='Blues', values_format='d', colorbar=False, ax=plt.gca())
                
                # Improve appearance
                plt.title(f'Confusion Matrix - {best_classifier_name} (Accuracy: {best_accuracy:.2f})')
                plt.grid(False)  # Remove grid lines
                plt.tight_layout()
                
                # Encode the figure
                cm_plot = encode_figure_to_base64(plt.gcf())
                plt.close()
            except Exception as e:
                print(f"Error creating confusion matrix: {str(e)}")
                cm_plot = None
        else:
            classifier_plot = None
            cm_plot = None
            best_classifier_name = None
            best_accuracy = None
    else:
        classifier_plot = None
        cm_plot = None
        best_classifier_name = None
        best_accuracy = None
    
    return {
        'n_events': len(valid_events),
        'event_ids': event_ids,
        'features': {
            'shape': X_features.shape,
            'feature_count': len(feature_names)
        },
        'plots': {
            'events': events_plot,
            'average_response': avg_response_plot,
            'regions': regions_plot,
            'classifier_comparison': classifier_plot,
            'confusion_matrix': cm_plot
        },
        'region_data': region_data,
        'best_classifier': best_classifier_name,
        'accuracy': best_accuracy
    }
# Modify the analyze_nirs_file function

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
        A dictionary containing the analysis results, including features and labels.
    """
    try:
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
            
            return direct_extract_features(raw_data, event_ids)
        else:
            return {'error': 'Failed to load NIRS data.'}
    except Exception as e:
        return {
            'error': f'Analysis failed: {str(e)}',
            'traceback': traceback.format_exc()
        }
