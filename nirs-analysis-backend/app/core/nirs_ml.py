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
from sklearn.model_selection import LeaveOneOut, GridSearchCV, learning_curve
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
        results_dict, y_true, y_pred, best_models = run_cross_validation(
            X_selected, labels, improved_classifiers, loo
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
        for name, clf in [('SVM', classifiers['SVM']), 
                         ('Ridge', classifiers['Ridge']),
                         ('RandomForest', classifiers['RandomForest'])]:
            if name in param_grids:
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
                    'best_score': grid.best_score_
                }
    
    return tuning_results

def run_cross_validation(X_selected, labels, classifiers, cv):
    """Run cross-validation for all classifiers"""
    results_dict = {}
    y_true = []
    y_pred = {}
    best_models = {}
    
    for name, clf in classifiers.items():
        print(f"Running cross-validation for {name}...")
        correct = 0
        y_pred[name] = []
        
        for train_idx, test_idx in cv.split(X_selected):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # Train and predict
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            correct += (pred[0] == y_test[0])
            
            if name == list(classifiers.keys())[0]:  # Only add once
                y_true.append(y_test[0])
            y_pred[name].append(pred[0])
            
            # Save the model from the last fold
            if train_idx[-1] == len(X_selected) - 2:  # Last fold
                best_models[name] = clf
        
        accuracy = correct / len(labels)
        results_dict[name] = accuracy
    
    return results_dict, y_true, y_pred, best_models

def create_classifier_comparison_plot(results_dict):
    """Create bar plot comparing classifier performance"""
    if not results_dict:
        return None
        
    fig = plt.figure(figsize=(10, 6))
    plt.bar(results_dict.keys(), results_dict.values())
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Classifier Comparison with Optimized Parameters')
    plt.ylim(0, 1)
    for i, (k, v) in enumerate(results_dict.items()):
        plt.text(i, v + 0.05, f'{v:.2f}', ha='center')
    plt.tight_layout()
    return encode_figure_to_base64(fig)

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
def generate_interpretation_metadata(feature_names, raw_data, brain_regions):
    """Generate metadata for feature interpretation"""
    # Create region descriptions
    region_descriptions = {
        'prefrontal': {
        'function': 'Funciones ejecutivas superiores',
        'examples': 'Planificación, toma de decisiones, inhibición de respuestas',
        'anatomical_areas': 'Corteza prefrontal dorsolateral, corteza orbitofrontal'
        },
        'central_frontal': {
        'function': 'Control motor y preparación motora',
        'examples': 'Planificación de movimientos, secuencias motoras',
        'anatomical_areas': 'Área motora suplementaria, córtex premotor'
        },
        'lateral_frontal': {
        'function': 'Procesamiento del lenguaje y memoria de trabajo',
        'examples': 'Tareas verbales, memoria verbal a corto plazo',
        'anatomical_areas': 'Área de Broca, corteza prefrontal ventrolateral'
        }
    }
    
    # Create channel mappings (S = source, D = detector)
    channel_mappings = {}
    for ch in raw_data.ch_names:
        parts = ch.split('_')
        if len(parts) >= 2:
            source = parts[0]  # e.g., S1, S2
            detector = parts[1]  # e.g., D1, D2
            wavelength = parts[2] if len(parts) > 2 else "unknown"
            
            channel_id = f"{source}_{detector}"
            channel_mappings[channel_id] = {
                'source_id': source,
                'detector_id': detector,
                'wavelength': wavelength,
                'anatomical_region': next(
                    (region for region, channels in brain_regions.items() 
                     if f"{source}_{detector}" in channels), 
                    'unknown'
                )
            }
    
    # Create feature explanations
    feature_explanations = {}
    for feature in feature_names:
        parts = feature.split('_')
        if len(parts) >= 3:
            region = parts[0]
            wavelength = parts[1]
            measure_type = '_'.join(parts[2:])
            
            # Create explanation based on feature components
            explanation = {
                'region': region,
                'region_function': region_descriptions.get(region, {}).get('function', 'Unknown function'),
                'wavelength': wavelength,
                'wavelength_meaning': '850nm - primarily oxygenated hemoglobin' if wavelength == '850' else '760nm - primarily deoxygenated hemoglobin',
                'measure_description': get_measure_description(measure_type)
            }
            
            feature_explanations[feature] = explanation
    
    # Package all interpretation data
    return {
        'region_descriptions': region_descriptions,
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