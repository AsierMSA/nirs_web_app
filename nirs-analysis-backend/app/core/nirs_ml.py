"""
This module contains functions for applying machine learning to NIRS data.
It includes methods for feature selection, classification, and visualization
of machine learning results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV, learning_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Import from nirs_processor to avoid circular imports
from .nirs_processor import encode_figure_to_base64

def apply_machine_learning(X_features, labels, feature_names):
    """
    Apply machine learning techniques to NIRS features.
    
    Parameters:
    ----------
    X_features : numpy.ndarray
        Feature matrix
    labels : numpy.ndarray
        Labels for classification
    feature_names : list
        Names of features for interpretation
        
    Returns:
    -------
    dict
        Dictionary containing ML results and visualizations
    """
    # Check if we have enough data
    if X_features.shape[0] <= 2 or len(np.unique(labels)) <= 1:
        return {'error': 'Insufficient data for machine learning analysis'}
    X_features = preprocess_features(X_features)
    # Feature selection
    k_features = min(X_features.shape[0]-1, X_features.shape[1])
    selector = SelectKBest(f_classif, k=k_features)
    X_selected = selector.fit_transform(X_features, labels)
    
    # Create feature importance visualization
    feature_importance_plot, top_features_by_fscore = create_feature_importance_plot(selector, feature_names)
    
    # Get top features for results
    # En la función apply_machine_learning, después de seleccionar los top features:

    # Get top features for results
    selected_indices = selector.get_support(indices=True)
    selected_features = top_features_by_fscore[:10] if top_features_by_fscore else []
    print (f"Top features selected: {selected_features}")

    
    # Define classifiers with optimized parameters
    improved_classifiers = {
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        'Ridge': RidgeClassifier(alpha=1.0),
        'SVM': SVC(kernel='linear', C=0.1, probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Initialize results
    results_dict = {}
    y_true = []
    y_pred = {}
    best_models = {}
    
    # Only attempt ML if we have enough samples
    if X_selected.shape[0] > 3:
        # Set up leave-one-out cross-validation
        loo = LeaveOneOut()
        
        # Perform hyperparameter tuning if enough samples
        tuning_results = perform_hyperparameter_tuning(X_selected, labels, improved_classifiers)
        
        # Create ensemble classifier with best models
        if X_selected.shape[0] >= 10:
            improved_classifiers['Ensemble'] = VotingClassifier(
                estimators=[
                    ('svm', improved_classifiers['SVM']),
                    ('rf', improved_classifiers['RandomForest']),
                    ('lda', improved_classifiers['LDA'])
                ],
                voting='soft'
            )
        
        # Run leave-one-out cross-validation
        results_dict, y_true, y_pred, best_models = run_block_cross_validation(
        X_selected, labels, improved_classifiers, loo, timestamps=None  
        )
        
        # Create comparison plot
        classifier_plot = create_classifier_comparison_plot(results_dict)
        
        # Get best classifier results
        best_classifier_name = max(results_dict, key=results_dict.get)
        best_accuracy = results_dict[best_classifier_name]
        
        # Create confusion matrix
        cm_plot = create_confusion_matrix_plot(y_true, y_pred[best_classifier_name], best_classifier_name, best_accuracy)
        
        # Create learning curve for best classifier
        learning_curve_plot = create_learning_curve_plot(
            best_models[best_classifier_name] if best_classifier_name in best_models else None,
            X_selected, labels, best_classifier_name
        )
    else:
        classifier_plot = None
        cm_plot = None
        best_classifier_name = None
        best_accuracy = None
        learning_curve_plot = None
        tuning_results = {}
    
    return {
        'top_features': selected_features,
        'best_classifier': best_classifier_name,
        'accuracy': best_accuracy,
        'plots': {
            'feature_importance': feature_importance_plot,
            'classifier_comparison': classifier_plot,
            'confusion_matrix': cm_plot,
            'learning_curve': learning_curve_plot
        },
        'params': tuning_results
    }

def create_feature_importance_plot(selector, feature_names):
    """Create feature importance visualization based on F-scores"""
    try:
        # Get feature scores and p-values
        scores = selector.scores_
        pvalues = selector.pvalues_
        
        # Get indices of selected features
        selected_indices = selector.get_support(indices=True)
        
        # Preparar lista para devolver las top features ordenadas por F-score
        top_features_by_fscore = []
        
        if len(selected_indices) > 0:
            # Create plot for top 15 features
            fig_feat, ax = plt.subplots(figsize=(10, 8))
            
            # Sort features by importance
            indices = np.argsort(scores)[-15:]
            
            # *** Log para la característica más importante ***
            if len(indices) > 0:
                # Guardar las top features ordenadas por F-score para devolverlas
                top_features_by_fscore = [feature_names[i] for i in reversed(indices)]
                
                most_important_idx = indices[-1]  # El último elemento es el más importante
                most_important_feature = feature_names[most_important_idx]
                
                print("\n" + "="*70)
                print(f"🥇 MOST IMPORTANT FEATURE: {most_important_feature}")
                print(f"   F-Score: {scores[most_important_idx]:.4f}, p-value: {pvalues[most_important_idx]:.4f}")
                
                # Parse feature components
                parts = most_important_feature.split('_')
                if len(parts) >= 3:
                    region = parts[0]
                    wavelength = parts[1]
                    measure_type = '_'.join(parts[2:])
                    print(f"   Region: {region}, Wavelength: {wavelength}nm, Measure: {measure_type}")
                print("="*70 + "\n")
            
            # Plot horizontal bar chart
            ax.barh(range(len(indices)), scores[indices], align='center')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices], fontsize=8)
            ax.set_xlabel('F-Score')
            ax.set_title('Top 15 Features by F-Score')
            
            # Add p-values as text
            for i, v in enumerate(indices):
                p_val_text = f"p={pvalues[v]:.3f}" if pvalues[v] >= 0.001 else "p<0.001"
                ax.text(scores[v] * 0.7, i, p_val_text, fontsize=7, va='center')
            
            plt.tight_layout()
            return encode_figure_to_base64(fig_feat), top_features_by_fscore
        return None, []
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")
        return None, []

def perform_hyperparameter_tuning(X_selected, labels, classifiers):
    """Perform hyperparameter tuning for selected classifiers"""
    tuning_results = {}
    
    if X_selected.shape[0] >= 10:
        print("Performing hyperparameter tuning...")
        # Define parameter grids for each classifier
        param_grids = {
            'SVM': {'C': [0.01, 0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
            'Ridge': {'alpha': [0.1, 1.0, 10.0]},
            'RandomForest': {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]}
        }
        
        # Only tune a subset of classifiers to save time
        for idx, (name, clf) in enumerate([('SVM', classifiers['SVM']), 
                                          ('Ridge', classifiers['Ridge']),
                                          ('RandomForest', classifiers['RandomForest'])]):
            if name in param_grids:
                print(f"  Tuning {name}... (progress: {(idx+1)*33}%)")
                
                # Use GridSearchCV with cross-validation
                grid = GridSearchCV(
                    clf, param_grids[name], cv=min(5, X_selected.shape[0]), 
                    scoring='accuracy', n_jobs=-1
                )
                grid.fit(X_selected, labels)
                
                # Update classifier with best parameters
                classifiers[name] = grid.best_estimator_
                tuning_results[name] = {
                    'best_params': grid.best_params_,
                    'best_score': grid.best_score_,
                    'tuning_complete': True,
                    'progress': (idx+1)*33  # Update progress percentage
                }
    
    return tuning_results
from scipy.signal import detrend

def preprocess_features(X):
    """Elimina tendencias temporales y normaliza por bloques"""
    X_detrended = detrend(X, axis=0, type='linear')
    
    # Normalización por bloques temporales
    scaler = StandardScaler()
    block_size = 30 
    for i in range(0, X.shape[0], block_size):
        block = slice(i, min(i+block_size, X.shape[0]))
        X_detrended[block] = scaler.fit_transform(X_detrended[block])
    
    return X_detrended

def run_block_cross_validation(X_selected, labels, classifiers, cv=None, timestamps=None):
    """Run block-based cross-validation to prevent temporal leakage"""
    results_dict = {}
    y_true = []
    y_pred = {}
    best_models = {}
    
    # Create temporal blocks if cv not provided
    if cv is None:
        n_blocks = min(5, len(labels)//2)
        cv = KFold(n_splits=n_blocks, shuffle=False)  # No shuffle preserves temporal order
    
    # Sort by timestamps if provided
    if timestamps is not None:
        time_sorted_indices = np.argsort(timestamps)
        X_selected = X_selected[time_sorted_indices]
        labels = labels[time_sorted_indices]
        print("Data sorted chronologically by timestamps for temporal validation")
    else:
        print("Warning: No timestamps provided, using data in original order")
    
    # Apply preprocessing to remove temporal trends
    X_processed = preprocess_features(X_selected)
    print("Preprocessing applied: detrending and block normalization")

    for name, clf in classifiers.items():
        print(f"Running block cross-validation for {name}...")
        scores = []
        y_pred[name] = []
        
        for train_idx, test_idx in cv.split(X_processed):
            X_train, X_test = X_processed[train_idx], X_processed[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # Train and predict
            clf.fit(X_train, y_train)
            
            # Store predictions
            pred = clf.predict(X_test)
            y_pred[name].extend(pred)
            
            # Only add true labels once
            if name == list(classifiers.keys())[0]:
                y_true.extend(y_test)
                
            score = clf.score(X_test, y_test)
            scores.append(score)
        results_dict[name] = np.mean(scores)
        
        # Store the trained classifier
        best_models[name] = clf
    
    return results_dict, y_true, y_pred, best_models
def create_classifier_comparison_plot(results_dict):
    """Create bar plot comparing classifier performance"""
    if not results_dict:
        return None
        
    fig = plt.figure(figsize=(10, 6))
    
    # Sort classifiers by performance for better visualization
    sorted_results = dict(sorted(results_dict.items(), key=lambda item: item[1], reverse=True))
    
    # Create bar colors based on performance
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_results)))
    
    # Plot bars with custom colors
    bars = plt.bar(range(len(sorted_results)), sorted_results.values(), color=colors)
    
    # Customize x-axis
    plt.xticks(range(len(sorted_results)), sorted_results.keys(), rotation=30, ha='right')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Classifier Comparison with Optimized Parameters')
    
    # Set y-axis limits with some padding
    plt.ylim(0, min(1.0, max(sorted_results.values()) * 1.2))
    
    # Add value labels on top of bars
    for i, (k, v) in enumerate(sorted_results.items()):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    # Highlight best classifier
    best_classifier = max(sorted_results.items(), key=lambda x: x[1])[0]
    best_idx = list(sorted_results.keys()).index(best_classifier)
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(1.5)
    
    # Add a legend
    plt.legend([bars[best_idx]], ['Best Performer'], loc='upper right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return encode_figure_to_base64(fig)

def validate_against_temporal_bias(X_features, labels, feature_names, timestamps=None):
    """Enhanced version with strict temporal controls"""
    # Ensure X_features is preprocessed before any analysis
    X_processed = preprocess_features(X_features)
    print("Applied preprocessing to remove temporal trends")
    
    # 1. Run regular analysis with preprocessed features
    regular_results = apply_machine_learning(X_processed, labels, feature_names)
    
    # 2. Control with temporal shift (circular shuffle)
    shift = len(labels)//2  # Mid-session shift
    shifted_labels = np.concatenate([labels[shift:], labels[:shift]])
    
    # 3. Control with chronologically sorted labels
    time_sorted_labels = np.sort(labels)
    
    # Run all controls
    controls = {
        'full_shuffle': np.random.permutation(labels),
        'temporal_shift': shifted_labels,
        'time_sorted': time_sorted_labels
    }
    
    control_results = {}
    for name, c_labels in controls.items():
        res = apply_machine_learning(X_processed, c_labels, feature_names)
        control_results[name] = res.get('accuracy', 0)
    
    # Calculate statistical significance
    real_accuracy = regular_results['accuracy'] if 'accuracy' in regular_results else 0
    shuffle_mean = np.mean([acc for acc in control_results.values()])
    p_value = sum(acc >= real_accuracy for acc in control_results.values()) / len(controls)
    
    return {
        'real_accuracy': real_accuracy,
        'shuffle_mean_accuracy': shuffle_mean,
        'controls': control_results,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'preprocessed': True
    }
def create_confusion_matrix_plot(y_true, y_pred, classifier_name, accuracy):
    """Create confusion matrix visualization"""
    try:
        # Create numeric-to-string label mapping for better readability
        unique_true = sorted(set(y_true))
        label_mapping = {label: f"Class {i+1}" for i, label in enumerate(unique_true)}
        
        # Convert labels
        y_true_labels = [label_mapping.get(y, f"Unknown-{y}") for y in y_true]
        y_pred_labels = [label_mapping.get(y, f"Unknown-{y}") for y in y_pred]
        
        # Get unique labels in order
        unique_labels = sorted(list(set(y_true_labels + y_pred_labels)))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=unique_labels)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        disp.plot(cmap='Blues', values_format='d', colorbar=False, ax=plt.gca())
        
        plt.title(f'Confusion Matrix - {classifier_name} (Accuracy: {accuracy:.2f})')
        plt.grid(False)
        plt.tight_layout()
        
        return encode_figure_to_base64(plt.gcf())
    except Exception as e:
        print(f"Error creating confusion matrix: {str(e)}")
        return None

def create_learning_curve_plot(clf, X_selected, labels, classifier_name):
    """Create learning curve visualization"""
    if clf is None or X_selected.shape[0] <= 5:
        return None
        
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate learning curve with appropriate train sizes
        train_sizes = np.linspace(0.1, 0.5, 5)  # Use smaller fractions
        n_samples = X_selected.shape[0]
        max_size = n_samples // 2  # Maximum size should be half the dataset
        
        # Convert fractions to absolute numbers, bounded by max_size
        train_sizes = [min(max_size, max(2, int(ts * n_samples))) for ts in train_sizes]
        train_sizes = sorted(set(train_sizes))  # Remove duplicates
        
        if len(train_sizes) >= 2:
            train_sizes, train_scores, test_scores = learning_curve(
                clf, X_selected, labels, 
                train_sizes=train_sizes,
                cv=min(3, X_selected.shape[0] // 2) if X_selected.shape[0] > 4 else 2,
                scoring='accuracy'
            )
            
            # Calculate mean and std
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            
            # Plot learning curve
            ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1, color='b')
            ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color='r')
            ax.plot(train_sizes, train_scores_mean, 'o-', color='b', label='Training score')
            ax.plot(train_sizes, test_scores_mean, 'o-', color='r', label='Cross-validation score')
            
            ax.set_xlabel('Training examples')
            ax.set_ylabel('Score')
            ax.set_title(f'Learning Curve for {classifier_name}')
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            return encode_figure_to_base64(fig)
        return None
    except Exception as e:
        print(f"Error generating learning curve: {str(e)}")
        return None
def generate_interpretation_metadata(feature_names, raw_data, brain_regions=None):
    """Generate metadata for feature interpretation based on individual channels"""
    # Crear diccionario vacío para brain_regions
    brain_regions = {}  # Ya no usamos brain_regions
    
    # Extraer canales únicos
    unique_channels = []
    for ch_name in raw_data.ch_names:
        parts = ch_name.split(' ')
        if len(parts) >= 1:
            channel = parts[0]  # Extraer solo el identificador S*_D*
            if channel not in unique_channels:
                unique_channels.append(channel)
    
    # Crear descripciones genéricas de canales
    channel_descriptions = {}
    for channel in unique_channels:
        if channel.startswith('S1') or channel.startswith('S2'):
            channel_descriptions[channel] = {
                'function': 'Funciones ejecutivas superiores (lóbulo frontal)',
                'examples': 'Planificación, toma de decisiones, inhibición de respuestas',
                'anatomical_areas': 'Corteza prefrontal'
            }
        elif channel.startswith('S3') or channel.startswith('S4'):
            channel_descriptions[channel] = {
                'function': 'Control motor y preparación motora (lóbulo frontal central)',
                'examples': 'Planificación de movimientos, secuencias motoras',
                'anatomical_areas': 'Área motora suplementaria'
            }
        elif channel.startswith('S5') or channel.startswith('S6'):
            channel_descriptions[channel] = {
                'function': 'Procesamiento de lenguaje y memoria de trabajo (lóbulo temporal)',
                'examples': 'Comprensión del lenguaje, memoria verbal',
                'anatomical_areas': 'Lóbulo temporal superior'
            }
        elif channel.startswith('S7') or channel.startswith('S8'):
            channel_descriptions[channel] = {
                'function': 'Integración sensorial y procesamiento espacial (lóbulo parietal)',
                'examples': 'Orientación espacial, atención selectiva',
                'anatomical_areas': 'Lóbulo parietal'
            }
        else:
            channel_descriptions[channel] = {
                'function': 'Actividad cerebral general',
                'examples': 'Procesamiento sensorial, cognitivo o motor',
                'anatomical_areas': 'Región cortical'
            }
    
    # Crear mapeo de canales
    channel_mappings = {}
    for ch in raw_data.ch_names:
        parts = ch.split(' ')
        if len(parts) >= 1:
            channel_parts = parts[0].split('_')
            if len(channel_parts) >= 2:
                source = channel_parts[0]  # e.g., S1, S2
                detector = channel_parts[1]  # e.g., D1, D2
                wavelength = parts[1] if len(parts) > 1 else "unknown"
                
                channel_id = f"{source}_{detector}"
                channel_mappings[channel_id] = {
                    'source_id': source,
                    'detector_id': detector,
                    'wavelength': wavelength,
                    'anatomical_region': 'individual_channel'  # Ya no usamos regiones predefinidas
                }
    
    # Crear explicaciones de características
    feature_explanations = {}
    for feature in feature_names:
        parts = feature.split('_')
        if len(parts) >= 3:
            channel_id = f"{parts[0]}_{parts[1]}"  # p.ej. S1_D1
            wavelength = parts[2]  # p.ej. 850
            measure_type = '_'.join(parts[3:]) if len(parts) > 3 else "unknown"
            
            # Crear explicación basada en componentes de la característica
            explanation = {
                'region': channel_id,
                'region_function': channel_descriptions.get(channel_id, {}).get('function', 'Unknown function'),
                'wavelength': wavelength,
                'wavelength_meaning': '850nm - primarily oxygenated hemoglobin' if wavelength == '850' else 
                                     '760nm - primarily deoxygenated hemoglobin',
                'measure_description': get_measure_description(measure_type)
            }
            
            feature_explanations[feature] = explanation
    
    # Empaquetar todos los datos de interpretación
    return {
        'region_descriptions': channel_descriptions,
        'channel_mappings': channel_mappings,
        'feature_explanations': feature_explanations,
        'event_descriptions': {
            'Exercise': 'Physical hand/finger movements recorded during the experiment',
            'Rest': 'Periods of inactivity between task trials',
            'Imagination': 'Mental imagery of movement without physical execution',
            'Baseline': 'Initial recording state before any stimulus presentation'
        }
    }
def get_measure_description(measure_type):
    """Get description for different measure types"""
    descriptions = {
        'early_mean': 'Average signal in the early response phase (1-4s after stimulus)',
        'middle_mean': 'Average signal in the middle response phase (5-10s after stimulus)',
        'late_mean': 'Average signal in the late response phase (11-15s after stimulus)',
        'slope_early': 'Rate of change between early and middle phases',
        'slope_late': 'Rate of change between middle and late phases',
        'amplitude': 'Maximum difference from baseline',
        'std': 'Overall variability in the signal'
    }
    return descriptions.get(measure_type, f"Measurement of {measure_type.replace('_', ' ')}")