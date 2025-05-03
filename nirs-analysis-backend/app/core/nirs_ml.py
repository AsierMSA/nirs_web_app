"""
This module contains functions for applying machine learning to NIRS data.
It includes methods for feature selection, classification, and visualization
of machine learning results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import clone
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE

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
    
    # --- Initial Preprocessing (Applied once before feature selection/tuning) ---
    # Detrend and scale the entire feature set initially.
    # Note: Scaling is re-applied within CV folds for robustness, but initial scaling
    # helps feature selection and tuning consistency. Detrending is often done once.
    scaler_initial = StandardScaler()
    X_features_detrended = detrend(X_features, axis=0, type='linear')
    X_features_scaled_initial = scaler_initial.fit_transform(X_features_detrended)
    print("Initial detrending and scaling applied.")
    # Use the initially processed features for feature selection and tuning input
    X_input_for_selection_tuning = X_features_scaled_initial
    # ---

    # Feature selection
    n_samples = X_input_for_selection_tuning.shape[0]
    n_features_available = X_input_for_selection_tuning.shape[1]
    k_features = min(30, n_features_available, n_samples - 1) # Ensure k < n_samples
    k_features = max(1, k_features) # Ensure k is at least 1
    n_classes = len(np.unique(labels))
    # --- cv_splits calculation (for tuning) ---
    cv_splits_tuning = min(5, n_samples // n_classes if n_classes > 0 else n_samples)
    cv_splits_tuning = max(2, cv_splits_tuning) # Need at least 2 splits
    # --- end calculation ---

    print(f"Selecting top {k_features} features using f_classif...")
    selector = SelectKBest(f_classif, k=k_features)
    # Fit selector on the initially processed data
    X_selected_for_tuning = selector.fit_transform(X_input_for_selection_tuning, labels)
    selected_indices = selector.get_support(indices=True)
    # selected_feature_names = [feature_names[i] for i in selected_indices] # Get names of selected features
    print(f"Shape after feature selection (for tuning): {X_selected_for_tuning.shape}")
    
    # Create feature importance visualization (based on initial selection)
    feature_importance_plot, top_features_by_fscore = create_feature_importance_plot(selector, feature_names) # Pass original names
    
    # Get top features for results display
    selected_features_display = top_features_by_fscore[:10] if top_features_by_fscore else []
    print (f"Top features selected (for display): {selected_features_display}")

    k_features = X_selected_for_tuning.shape[1]
    input_shape_cnn = (k_features, 1)
    num_classes_cnn = len(np.unique(labels))
    # Define base classifiers
    base_classifiers = {
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'Ridge': RidgeClassifier(max_iter=2000), 
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced'), 
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'), 
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'CNN_1D': KerasClassifier(
                model=create_cnn_model,input_shape=input_shape_cnn,
                num_classes=num_classes_cnn,
                filters=32,
                kernel_size=3,
                dropout_rate=0.4, 
                epochs=60, 
                batch_size=16, 
                verbose=0 
            )
    }
    # Perform hyperparameter tuning using the selected features from the initial processing step
    tuned_classifiers, tuning_results = perform_hyperparameter_tuning(
        X_selected_for_tuning, labels, base_classifiers, cv_splits=cv_splits_tuning
    )

    # Initialize results
    results_dict = {}
    y_true_final_cv = [] # Renamed to avoid confusion
    y_pred_final_cv = {} # Renamed to avoid confusion
    best_models_cv = {} # Renamed to avoid confusion
    classifier_plot = None
    cm_plot = None
    best_classifier_name = None
    best_accuracy = None
    learning_curve_plot = None

    # Determine CV splits for the main cross-validation run
    cv_splits_main = min(5, n_samples // n_classes if n_classes > 0 else n_samples)
    cv_splits_main = max(2, cv_splits_main)

    # Only attempt ML if we have enough samples *before* selection
    if X_features.shape[0] > cv_splits_main:
        # Create ensemble classifier with *tuned* models if enough samples and base models exist
        if X_features.shape[0] >= 10 and all(name in tuned_classifiers for name in ['SVM', 'RandomForest', 'LDA']):
             # Ensure base models for ensemble are actually present after tuning
             estimators_for_ensemble = []
             if 'SVM' in tuned_classifiers: estimators_for_ensemble.append(('svm', tuned_classifiers['SVM']))
             if 'RandomForest' in tuned_classifiers: estimators_for_ensemble.append(('rf', tuned_classifiers['RandomForest']))
             if 'LDA' in tuned_classifiers: estimators_for_ensemble.append(('lda', tuned_classifiers['LDA']))

             if len(estimators_for_ensemble) >= 2: # Need at least 2 estimators for VotingClassifier
                 tuned_classifiers['Ensemble'] = VotingClassifier(
                     estimators=estimators_for_ensemble,
                     voting='soft' # Soft voting often works well if classifiers are calibrated (probability=True)
                 )
             else:
                 print("Not enough base models available after tuning to create Ensemble.")


        # --- Run cross-validation ---
        # Pass the *original* X_features (detrended but not scaled/selected yet)
        # The run_block_cross_validation function handles scaling, SMOTE, and uses the *tuned* classifiers.
        # It also needs the *selector* fitted earlier to apply feature selection within each fold.
        results_dict, y_true_final_cv, y_pred_final_cv, best_models_cv = run_block_cross_validation(
            X_features_detrended, # Pass detrended data, scaling/selection happens inside CV
            labels,
            tuned_classifiers,
            selector, # Pass the fitted selector
            cv=None, # Let the function determine splits
            timestamps=None
        )
        # ---

        if results_dict: # Check if CV produced results
            # Create comparison plot
            classifier_plot = create_classifier_comparison_plot(results_dict)

            # Get best classifier results
            best_classifier_name = max(results_dict, key=results_dict.get)
            best_accuracy = results_dict[best_classifier_name]

            # --- Create confusion matrix using the final filtered results from CV ---
            if best_classifier_name in y_pred_final_cv:
                cm_plot = create_confusion_matrix_plot(
                    y_true_final_cv, # Use the filtered true labels returned by CV
                    y_pred_final_cv[best_classifier_name], # Use the filtered predictions for the best model
                    best_classifier_name,
                    best_accuracy
                )
            else:
                print(f"Warning: No predictions found for the best classifier '{best_classifier_name}' after CV.")
            # ---

            # Create learning curve for best classifier
            best_model_instance = best_models_cv.get(best_classifier_name)
            if best_model_instance:
                 # We need to pass the data *as it was used for tuning* to learning_curve
                 # because the best_model_instance expects data with 'k_features' dimensions.
                 learning_curve_plot = create_learning_curve_plot(
                     best_model_instance, # The final tuned model instance
                     X_selected_for_tuning, # Data used for tuning (already selected)
                     labels,
                     best_classifier_name
                 )
            else:
                 print(f"Warning: No best model instance found for '{best_classifier_name}' to generate learning curve.")

        else:
             print("Cross-validation did not produce any results.")
             # Reset potentially assigned values if CV failed
             classifier_plot = None
             best_classifier_name = None
             best_accuracy = None
             cm_plot = None
             learning_curve_plot = None

    else:
        print("Not enough samples for cross-validation after initial checks.")
        # Ensure results are empty/None if CV is skipped
        classifier_plot = None
        cm_plot = None
        best_classifier_name = None
        best_accuracy = None
        learning_curve_plot = None
        tuning_results = {} # Clear tuning results as well if CV wasn't run
    
    return {
        'top_features': selected_features_display, # Use the display list
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
        
        # Prepare list to return the top features sorted by F-score
        top_features_by_fscore = []
        
        if len(selected_indices) > 0:
            # Create plot for top 15 features
            fig_feat, ax = plt.subplots(figsize=(10, 8))
            
            # Sort features by importance
            indices = np.argsort(scores)[-15:]
            
            # *** Log for the most important feature ***
            if len(indices) > 0:
                # Save the top features sorted by F-score to return them
                top_features_by_fscore = [feature_names[i] for i in reversed(indices)]
                
                most_important_idx = indices[-1]  # The last element is the most important
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

def perform_hyperparameter_tuning(X_selected, labels, classifiers, cv_splits): # Added cv_splits argument
    """Perform more extensive hyperparameter tuning."""
    tuning_results = {}

    # Add Gradient Boosting to classifiers if not present (already done in apply_machine_learning)
    # if 'GradientBoosting' not in classifiers:
    #     classifiers['GradientBoosting'] = GradientBoostingClassifier(random_state=42)

    # Define more comprehensive parameter grids
    param_grids = {
        'SVM': {
            'C': [0.01, 0.1, 1, 10], 
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1], 
            'degree': [2, 3],
            'class_weight': ['balanced'] 
        },
        'Ridge': {
            'alpha': [0.1, 1.0, 10.0, 100.0],
            'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'max_iter': [2000, 3000] 
         },
        'RandomForest': {
            'n_estimators': [50, 100, 150], 
            'max_depth': [5, 10, 15], 
            'min_samples_split': [5, 10, 15], 
            'min_samples_leaf': [3, 5, 7], 
            'class_weight': ['balanced', 'balanced_subsample'] 
        },
        'LDA': [
            {'solver': ['svd']},
            {'solver': ['lsqr', 'eigen'], 'shrinkage': ['auto', 0.1, 0.5]} 
        ],
        'GradientBoosting': {
             'n_estimators': [50, 100],
             'learning_rate': [0.01, 0.1],
             'max_depth': [3, 5] 
        }
    }

    inner_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42) # Use passed cv_splits

    n_samples = X_selected.shape[0]
    if n_samples >= cv_splits * 2: # Check if enough samples for tuning
        print(f"Performing hyperparameter tuning with GridSearchCV (inner CV splits={cv_splits})...")
        tuned_classifiers = {}
        # --- Reshape data ONCE if CNN is being tuned ---
        X_selected_cnn = None
        num_classifiers_to_tune = len([name for name in classifiers if name in param_grids])
        progress_counter = 0

        for name, clf in classifiers.items():
            if name in param_grids:
                current_param_grid = param_grids[name] # Get grid for current classifier

                # --- Select correct data shape for fitting GridSearch ---
                X_fit_grid = X_selected # Default shape
                is_cnn_tuning = (name == 'CNN_1D' and X_selected_cnn is not None)
                if is_cnn_tuning:
                    X_fit_grid = X_selected_cnn # Use reshaped data for CNN
                # ---

                progress_counter += 1
                print(f"  Tuning {name}... (progress: {int((progress_counter/num_classifiers_to_tune)*100)}%)")
                try:
                    # Pass the correct param_grid (might be list for LDA)
                    grid = GridSearchCV(
                        clf, current_param_grid, cv=inner_cv,
                        scoring='accuracy', n_jobs=-1, error_score='raise', refit=True
                    )
                    # Fit with potentially reshaped data
                    grid.fit(X_fit_grid, labels)

                    tuned_classifiers[name] = grid.best_estimator_
                    tuning_results[name] = {
                        'best_params': grid.best_params_,
                        'best_score': grid.best_score_,
                        'tuning_complete': True,
                        'progress': int((progress_counter/num_classifiers_to_tune)*100)
                    }
                    print(f"    Best params for {name}: {grid.best_params_}")
                    print(f"    Best score for {name}: {grid.best_score_:.4f}")
                except Exception as e:
                    print(f"    ERROR tuning {name}: {e}. Using default parameters.")
                    import traceback
                    traceback.print_exc() # Print full traceback for tuning errors
                    tuned_classifiers[name] = clone(clf) # Use a clone of the original default
                    tuning_results[name] = {'error': str(e), 'tuning_complete': False}
            else:
                # Keep classifiers without grids as they are (use a clone)
                tuned_classifiers[name] = clone(clf)

        print("Hyperparameter tuning finished.")
        return tuned_classifiers, tuning_results
    else:
         print("Not enough samples for reliable hyperparameter tuning. Using default parameters.")
         # Ensure CNN is cloned correctly if not tuned
         return {name: clone(clf) for name, clf in classifiers.items()}, {}
    
def add_gaussian_noise(X, noise_level=0.02):
    """
    Adds Gaussian noise to features.

    Args:
        X (np.ndarray): Feature matrix (samples x features).
        noise_level (float): Standard deviation of the noise relative to feature std dev.

    Returns:
        np.ndarray: Feature matrix with added noise.
    """
    if X.shape[0] == 0: # Handle empty input
        return X
    # Calculate std dev per feature, adding epsilon to avoid division by zero
    std_dev = np.std(X, axis=0) + 1e-6
    noise = np.random.normal(0, noise_level * std_dev, X.shape)
    return X + noise

import tensorflow as tf
from tensorflow import keras
from keras import layers
from scikeras.wrappers import KerasClassifier
def create_cnn_model(input_shape, num_classes, filters=32, kernel_size=3, dropout_rate=0.4):
    """
    Creates a simple 1D CNN model suitable for sequence-like feature data.

    Args:
        input_shape (tuple): Shape of the input features (e.g., (num_features, 1)).
        num_classes (int): Number of output classes.
        filters (int): Number of filters in Conv1D layers.
        kernel_size (int): Size of the convolution kernel.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        keras.Model: Compiled Keras model.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            # Conv Block 1
            layers.Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same"),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(dropout_rate),
            # Conv Block 2
            layers.Conv1D(filters=filters * 2, kernel_size=kernel_size, activation="relu", padding="same"),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(dropout_rate),
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(64, activation="relu"), 
            layers.Dropout(dropout_rate),
            # Output layer
            layers.Dense(num_classes, activation="softmax"), 
        ]
    )
    # Compile the model
    # ALWAYS use sparse_categorical_crossentropy for integer labels and softmax output
    loss_function = 'sparse_categorical_crossentropy'
    model.compile(optimizer='adam',
                  loss=loss_function,
                  metrics=['accuracy'])
    return model
from scipy.signal import detrend

def preprocess_features(X_train, X_test=None):
    """Detrend and scale features. Fit scaler only on training data."""
    # Detrend train data
    X_train_detrended = detrend(X_train, axis=0, type='linear')

    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_detrended)

    X_test_scaled = None
    if X_test is not None:
        # Detrend test data
        X_test_detrended = detrend(X_test, axis=0, type='linear')
        # Transform test data using the scaler fitted on train data
        X_test_scaled = scaler.transform(X_test_detrended)

    return X_train_scaled, X_test_scaled, scaler
def run_block_cross_validation(X_detrended, labels, classifiers, selector, cv=None, timestamps=None, cnn_input_shape=None):
    results_dict = {}
    y_true_all = []
    y_pred_all = {name: [] for name in classifiers}
    trained_models = {name: [] for name in classifiers} # Store models from each fold

    # Determine CV splits if not provided
    if cv is None:
        n_samples = X_detrended.shape[0]
        n_classes = len(np.unique(labels))
        # Ensure labels is integer array
        labels_int = np.array(labels, dtype=int)
        min_samples_per_class = 0 # Initialize

        if n_classes > 1 and len(labels_int) > 0:
            unique_labels, counts = np.unique(labels_int, return_counts=True)
            if len(counts) > 0:
                min_samples_per_class = np.min(counts)
            else: # Should not happen if n_classes > 1 and len > 0, but for safety
                min_samples_per_class = 0
        elif n_samples > 0: # Only one class or empty labels
             min_samples_per_class = n_samples

        # Determine n_splits, ensuring it's at least 2 and not more than samples or min class count
        n_splits = 5 # Default target
        if n_classes > 1 and min_samples_per_class > 0: # Check min_samples > 0
            n_splits = min(n_splits, min_samples_per_class)
        elif n_samples > 0: # Only one class or calculation failed, base on n_samples
             # Use n_samples // 2 for single class, ensure at least 2 samples per split
             n_splits = min(n_splits, n_samples // 2 if n_samples >= 4 else n_samples)

        n_splits = max(2, n_splits) # Must be at least 2

        # Final check if n_splits is feasible (e.g., n_samples=3, n_splits becomes 3 initially, adjust to 2)
        if n_samples < n_splits:
             # If n_samples is 2 or 3, n_splits should be 2. If n_samples is 1, it won't reach here.
             n_splits = n_samples if n_samples < 2 else 2

        print(f"Attempting StratifiedKFold for CV. Calculated n_splits={n_splits}, min_samples_per_class={min_samples_per_class}, n_samples={n_samples}")
        # --- (Rest of the try/except block for CV setup remains the same) ---
        try:
            # Ensure n_splits is valid before creating StratifiedKFold
            if n_splits < 2:
                raise ValueError(f"Calculated n_splits ({n_splits}) is less than 2.")
            # Check if stratification is possible
            if n_classes > 1 and n_splits > min_samples_per_class:
                 print(f"Warning: Cannot use StratifiedKFold with n_splits={n_splits} > min_samples_per_class={min_samples_per_class}. Falling back.")
                 raise ValueError("n_splits > number of members in the least populated class.") # Trigger fallback

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            _ = list(cv.split(X_detrended, labels_int)) # Test split feasibility
            print(f"Using StratifiedKFold with {n_splits} splits for CV.")

        except ValueError as e:
            print(f"Warning: StratifiedKFold failed ({e}). Falling back to KFold.")
            # Fallback logic remains the same, using the calculated n_splits
            n_splits_kfold = min(n_splits, n_samples)
            n_splits_kfold = max(2, n_splits_kfold)
            if n_samples < n_splits_kfold: # Final check for KFold
                 n_splits_kfold = n_samples
            if n_splits_kfold < 2 and n_samples >= 2: # Handle edge case where n_samples=2 or 3
                 n_splits_kfold = 2
            elif n_samples < 2:
                 print("Error: Cannot perform KFold with less than 2 samples.")
                 return {}, [], {}, {} # Cannot proceed

            cv = KFold(n_splits=n_splits_kfold, shuffle=True, random_state=42)
            print(f"Using KFold with {n_splits_kfold} splits.")
        except Exception as e_gen: # Catch other potential errors
             print(f"Error setting up CV: {e_gen}. Cannot proceed.")
             return {}, [], {}, {}


    fold_num = 0
    try:
        actual_splits = cv.get_n_splits(X_detrended, labels) # Get actual number of splits
    except Exception as e:
        print(f"Error getting number of splits: {e}. Cannot proceed with CV.")
        return {}, [], {}, {} # Return empty results

    # --- Main CV Loop START ---
    for train_idx, test_idx in cv.split(X_detrended, labels):
        fold_num += 1
        print(f"  Processing Fold {fold_num}/{actual_splits}...")

        # Ensure indices are valid and splits are not empty
        if len(train_idx) == 0 or len(test_idx) == 0:
            print(f"    Skipping Fold {fold_num}: Empty train or test set.")
            continue

        X_train_fold_det, X_test_fold_det = X_detrended[train_idx], X_detrended[test_idx]
        y_train_fold, y_test_fold = labels[train_idx], labels[test_idx]

        # --- Preprocessing inside the loop ---
        # 1. Scaling: Fit scaler ONLY on training data for this fold
        scaler_fold = StandardScaler()
        X_train_scaled = scaler_fold.fit_transform(X_train_fold_det)
        X_test_scaled = scaler_fold.transform(X_test_fold_det)

        # 2. Feature Selection: Transform using the selector fitted *outside* the loop
        try:
            X_train_selected = selector.transform(X_train_scaled)
            X_test_selected = selector.transform(X_test_scaled)
            print(f"    Scaling and Feature Selection applied to fold {fold_num}.")
        except ValueError as e:
             # Handle cases where transform might fail (e.g., different number of features if selector was refit)
             print(f"    ERROR applying feature selection transform in fold {fold_num}: {e}. Skipping fold.")
             # Append NaNs for this fold's predictions
             y_true_all.extend(y_test_fold)
             for name in classifiers: y_pred_all[name].extend([np.nan] * len(y_test_fold))
             continue
        except Exception as e:
            print(f"    UNEXPECTED ERROR during feature selection in fold {fold_num}: {e}. Skipping fold.")
            y_true_all.extend(y_test_fold)
            for name in classifiers: y_pred_all[name].extend([np.nan] * len(y_test_fold))
            continue


        # --- Check for SMOTE validity ---
        unique_train_labels, counts_train_labels = np.unique(y_train_fold, return_counts=True)
        n_classes_fold = len(np.unique(labels)) # Use overall n_classes for check

        # Check if train fold contains at least 2 classes if overall there are multiple classes
        can_smote = True
        if n_classes_fold > 1 and len(unique_train_labels) < 2:
            print(f"    Skipping SMOTE for Fold {fold_num}: Training split has only {len(unique_train_labels)} class(es), need at least 2 for SMOTE.")
            can_smote = False
        elif n_classes_fold > 1 and len(unique_train_labels) < n_classes_fold:
             print(f"    Warning for Fold {fold_num}: Training split does not contain all classes ({len(unique_train_labels)}/{n_classes_fold}). SMOTE might be less effective.")
             # Proceed with SMOTE if at least 2 classes are present

        # --- Apply SMOTE ---
        X_train_resampled, y_train_resampled = X_train_selected, y_train_fold # Default to non-resampled
        if can_smote and n_classes_fold > 1: # Only apply if more than 1 class overall and train fold has >= 2 classes
            try:
                min_class_count = np.min(counts_train_labels)
                smote_k_neighbors = min(5, min_class_count - 1)

                if smote_k_neighbors >= 1:
                    smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
                    print(f"    Applying SMOTE (k={smote_k_neighbors}) to training data of fold {fold_num}...")
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train_fold)
                    print(f"    Original train shape: {X_train_selected.shape}, Resampled train shape: {X_train_resampled.shape}")
                else:
                     print(f"    Skipping SMOTE for fold {fold_num}: Not enough samples in the smallest class ({min_class_count}) for k_neighbors > 0.")

            except Exception as e:
                 print(f"    Error applying SMOTE in fold {fold_num}: {e}. Using original training data for this fold.")
                 # X_train_resampled, y_train_resampled are already set to original selected data
        # --- End SMOTE ---

        # --- Data AugmentatioZzEn: Add Gaussian Noise ---
        # Apply noise to the (potentially SMOTE-resampled) training data
        # Create one noisy copy for each sample in the current training set
        print(f"    Applying Gaussian Noise Augmentation (noise_level=0.02)...")
        X_train_noisy = add_gaussian_noise(X_train_resampled, noise_level=0.02)

        # Combine original (resampled) data with noisy data
        X_train_augmented = np.vstack((X_train_resampled, X_train_noisy))
        y_train_augmented = np.concatenate((y_train_resampled, y_train_resampled)) # Duplicate labels for noisy copies

        print(f"    Shape after augmentation: {X_train_augmented.shape}")
        # --- End Data Augmentation ---


        # Append true labels for this fold (only done once per fold)
        y_true_all.extend(y_test_fold)

        # --- Train and Predict with each classifier ---
        # !!! USE AUGMENTED DATA FOR TRAINING !!!
        for name, clf_template in classifiers.items():
            # Ensure the classifier instance exists (might be None if tuning failed)
            if clf_template is None:
                print(f"    Skipping {name} in fold {fold_num}: Classifier instance is None.")
                y_pred_all[name].extend([np.nan] * len(y_test_fold))
                continue

            clf = clone(clf_template)
            try:
                # --- Reshape data if the classifier is CNN ---
                # Use the AUGMENTED training data for fitting
                X_train_fit = X_train_augmented
                X_test_predict = X_test_selected # Test data remains unchanged

                # Check if the classifier is a KerasClassifier (for CNN)
                is_cnn = isinstance(clf, KerasClassifier)

                if is_cnn and cnn_input_shape:
                    # Reshape AUGMENTED training data and original test data
                    n_features = cnn_input_shape[0]
                    if X_train_fit.shape[1] == n_features:
                         X_train_fit = X_train_fit.reshape((X_train_fit.shape[0], n_features, 1))
                    else:
                         print(f"    Warning: Mismatch in feature count for CNN training data reshape in fold {fold_num}. Expected {n_features}, got {X_train_fit.shape[1]}. Skipping reshape.")
                         # Potentially skip this classifier for this fold or handle differently
                         # For now, we'll let it potentially fail in fit() if shape is wrong

                    if X_test_predict.shape[1] == n_features:
                         X_test_predict = X_test_predict.reshape((X_test_predict.shape[0], n_features, 1))
                    else:
                         print(f"    Warning: Mismatch in feature count for CNN test data reshape in fold {fold_num}. Expected {n_features}, got {X_test_predict.shape[1]}. Skipping reshape.")
                         # Potentially skip prediction or handle differently

                    print(f"    Reshaped data for {name} (CNN) to {X_train_fit.shape if X_train_fit.ndim == 3 else 'original shape due to mismatch'}")
                # --- End Reshape ---

                # Train on the AUGMENTED (and potentially reshaped) training data
                # !!! Pass the augmented labels !!!
                clf.fit(X_train_fit, y_train_augmented)

                # Predict on the original processed (scaled, selected) and RESHAPED test data
                pred = clf.predict(X_test_predict)
                y_pred_all[name].extend(pred)
                trained_models[name].append(clf) # Store the trained model for this fold
            except Exception as e:
                 print(f"    ERROR training/predicting {name} in fold {fold_num}: {e}")
                 import traceback
                 traceback.print_exc() # Print detailed traceback for debugging
                 y_pred_all[name].extend([np.nan] * len(y_test_fold))
                 trained_models[name].append(None) # Add None to keep list lengths consistent

    # --- Main CV Loop END ---


    # --- Calculations AFTER the loop ---
    final_best_models = {} # Store the single best model instance for each classifier type (e.g., from last fold)
    print("\nCalculating final accuracies...")
    for name in classifiers:
        # Check if predictions exist for this classifier
        if name not in y_pred_all or not y_pred_all[name]:
             print(f"  Accuracy for {name}: 0.0000 (No predictions generated)")
             results_dict[name] = 0.0
             continue

        # Filter out NaNs if they were added due to errors/skips
        valid_indices = [i for i, p in enumerate(y_pred_all[name]) if p is not None and not np.isnan(p)]

        if y_true_all and valid_indices and len(valid_indices) > 0: # Check if there are any valid predictions
             y_true_filtered = [y_true_all[i] for i in valid_indices]
             y_pred_filtered = [y_pred_all[name][i] for i in valid_indices]

             # Ensure lengths match after filtering (should always if logic is correct)
             if len(y_true_filtered) == len(y_pred_filtered):
                 try:
                     accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
                     results_dict[name] = accuracy
                     print(f"  Accuracy for {name}: {accuracy:.4f}")
                     # Store the last valid trained model instance for this classifier type
                     valid_trained_fold_models = [m for m in trained_models.get(name, []) if m is not None]
                     if valid_trained_fold_models:
                         final_best_models[name] = valid_trained_fold_models[-1] # Store the model from the last valid fold
                 except Exception as e:
                      print(f"  Error calculating accuracy for {name}: {e}")
                      results_dict[name] = 0.0
             else:
                  print(f"  Accuracy calculation skipped for {name}: Mismatched lengths after filtering NaNs ({len(y_true_filtered)} vs {len(y_pred_filtered)}).")
                  results_dict[name] = 0.0
        else:
             # Handle cases with no valid predictions or if y_true_all is empty
             results_dict[name] = 0.0
             print(f"  Accuracy for {name}: 0.0000 (No valid predictions or all folds skipped/failed)")


    # Find best classifier based on calculated accuracy
    best_classifier_name_cv = max(results_dict, key=results_dict.get) if results_dict else None

    # Prepare final outputs: Filter true labels and predictions *specifically for the best model*
    y_true_final_output = []
    y_pred_final_output = {} # Dictionary containing only the best model's predictions

    if best_classifier_name_cv and y_pred_all.get(best_classifier_name_cv):
        # Filter NaNs specifically for the best classifier's predictions
        valid_indices_best = [i for i, p in enumerate(y_pred_all[best_classifier_name_cv]) if p is not None and not np.isnan(p)]
        if valid_indices_best:
            # Use these indices to get the corresponding true labels
            y_true_final_output = [y_true_all[i] for i in valid_indices_best]
            # Get the filtered predictions for the best model
            best_y_pred_filtered = [y_pred_all[best_classifier_name_cv][i] for i in valid_indices_best]
            # Store only the best model's filtered predictions in the output dict
            y_pred_final_output = {best_classifier_name_cv: best_y_pred_filtered}
        else:
             # Handle case where even the best classifier had no valid predictions
             print(f"Warning: Best classifier '{best_classifier_name_cv}' had no valid predictions.")
             y_true_final_output = [] # Keep it empty
             y_pred_final_output = {}
    else:
        # Handle case where no classifier produced results or y_true_all is empty
        print("Warning: No best classifier found or no predictions available.")
        y_true_final_output = [] # Keep it empty
        y_pred_final_output = {}


    # --- Correctly indented return statement ---
    # Return:
    # 1. Dictionary of accuracies for all classifiers.
    # 2. List of true labels corresponding to the *best* classifier's valid predictions.
    # 3. Dictionary containing only the *best* classifier's valid predictions.
    # 4. Dictionary of the final trained model instances (one per classifier type).
    return results_dict, y_true_final_output, y_pred_final_output, final_best_models

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
    X_processed, _, _ = preprocess_features(X_features)
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
    # --- Add check for empty inputs ---
    if not isinstance(y_true, (list, np.ndarray)) or not isinstance(y_pred, (list, np.ndarray)) or len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        print(f"Warning: Cannot create confusion matrix for {classifier_name}. Invalid or empty input labels/predictions.")
        return None
    # ---

    try:
        # Create numeric-to-string label mapping for better readability
        # Use unique labels present in both true and predicted, plus any potential original classes
        # This handles cases where a class might not appear in y_pred for the best model
        all_possible_labels_numeric = sorted(list(set(y_true))) # Assuming y_true contains all possible classes eventually
        label_mapping = {label: f"Class {i+1}" for i, label in enumerate(all_possible_labels_numeric)}

        # Convert labels using the mapping
        y_true_labels = [label_mapping.get(y, f"Unknown-{y}") for y in y_true]
        y_pred_labels = [label_mapping.get(y, f"Unknown-{y}") for y in y_pred]

        # Define display labels based on the full mapping
        display_labels = [label_mapping[label] for label in all_possible_labels_numeric]

        # Create confusion matrix using the display labels to ensure all classes are shown
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=display_labels)

        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 6)) # Adjusted size slightly
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap='Blues', values_format='d', colorbar=False, ax=ax) # Pass ax

        ax.set_title(f'Confusion Matrix - {classifier_name} (Accuracy: {accuracy:.2f})')
        plt.grid(False)
        plt.tight_layout()

        encoded_fig = encode_figure_to_base64(fig)
        plt.close(fig) # Close figure after encoding
        return encoded_fig
    except Exception as e:
        print(f"Error creating confusion matrix for {classifier_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Ensure figure is closed if error occurs
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)
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
    # Create empty dictionary for brain_regions
    brain_regions = {}  # We no longer use brain_regions
    
    # Extract unique channels
    unique_channels = []
    for ch_name in raw_data.ch_names:
        parts = ch_name.split(' ')
        if len(parts) >= 1:
            channel = parts[0]  # Extract only the S*_D* identifier
            if channel not in unique_channels:
                unique_channels.append(channel)
    
    # Create generic channel descriptions
    channel_descriptions = {}
    for channel in unique_channels:
        if channel.startswith('S1') or channel.startswith('S2'):
            channel_descriptions[channel] = {
                'function': 'Higher executive functions (frontal lobe)',
                'examples': 'Planning, decision making, response inhibition',
                'anatomical_areas': 'Prefrontal cortex'
            }
        elif channel.startswith('S3') or channel.startswith('S4'):
            channel_descriptions[channel] = {
                'function': 'Motor control and motor preparation (central frontal lobe)',
                'examples': 'Movement planning, motor sequences',
                'anatomical_areas': 'Supplementary motor area'
            }
        elif channel.startswith('S5') or channel.startswith('S6'):
            channel_descriptions[channel] = {
                'function': 'Language processing and working memory (temporal lobe)',
                'examples': 'Language comprehension, verbal memory',
                'anatomical_areas': 'Superior temporal lobe'
            }
        elif channel.startswith('S7') or channel.startswith('S8'):
            channel_descriptions[channel] = {
                'function': 'Sensory integration and spatial processing (parietal lobe)',
                'examples': 'Spatial orientation, selective attention',
                'anatomical_areas': 'Parietal lobe'
            }
        else:
            channel_descriptions[channel] = {
                'function': 'General brain activity',
                'examples': 'Sensory, cognitive, or motor processing',
                'anatomical_areas': 'Cortical region'
            }
    
    # Create channel mappings
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
                    'anatomical_region': 'individual_channel'  # We no longer use predefined regions
                }
    
    # Create feature explanations
    feature_explanations = {}
    for feature in feature_names:
        parts = feature.split('_')
        if len(parts) >= 3:
            channel_id = f"{parts[0]}_{parts[1]}"  # e.g. S1_D1
            wavelength = parts[2]  # e.g. 850
            measure_type = '_'.join(parts[3:]) if len(parts) > 3 else "unknown"
            
            # Create explanation based on feature components
            explanation = {
                'region': channel_id,
                'region_function': channel_descriptions.get(channel_id, {}).get('function', 'Unknown function'),
                'wavelength': wavelength,
                'wavelength_meaning': '850nm - primarily oxygenated hemoglobin' if wavelength == '850' else 
                                     '760nm - primarily deoxygenated hemoglobin',
                'measure_description': get_measure_description(measure_type)
            }
            
            feature_explanations[feature] = explanation
    
    # Package all interpretation data
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