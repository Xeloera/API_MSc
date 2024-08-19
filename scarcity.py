
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from joblib import dump

# Function to load, concatenate, and save arrays for given names
def load_and_save_permutations(names, array_type, results_dir='results'):
    main_array_path = os.path.join(results_dir, f'clip_{array_type}.npy')
    
    if not os.path.exists(main_array_path):
        raise FileNotFoundError(f"Main array not found at path: {main_array_path}")
    
    main_array = np.load(main_array_path)
    arrays = []
    
    for name in names:
        array_path = os.path.join(results_dir, f'clip_{array_type}_{name}.npy')
        if os.path.exists(array_path):
            arrays.append(np.load(array_path))
        else:
            raise FileNotFoundError(f"Augmentation array not found at path: {array_path}")
    
    merged_array = np.concatenate(arrays, axis=0)
    filename = '_and_'.join(names)
    save_path = os.path.join(results_dir, f'clip_{array_type}_{filename}_final.npy')
    np.save(save_path, merged_array)
    print(f"Merged array saved to {save_path}")
    return save_path

def summarize_data(embeddings, labels, data_type=""):
    print(f"{data_type} Data Summary")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels distribution: {np.sum(labels, axis=0)}")

def combine_embeddings(main_embeddings, main_labels, aug_embeddings, aug_labels):
    combined_embeddings = []
    combined_labels = []

    if aug_embeddings is not None:
        # Create a set of hashable embeddings to check for duplicates
        main_set = set(map(tuple, main_embeddings))

        for i, emb in enumerate(aug_embeddings):
            if tuple(emb) not in main_set:
                combined_embeddings.append(emb)
                combined_labels.append(aug_labels[i])

    # Convert to numpy arrays and ensure they are at least 2D
    combined_embeddings = np.array(combined_embeddings)
    combined_labels = np.array(combined_labels)

    if combined_embeddings.ndim == 1:
        combined_embeddings = combined_embeddings.reshape(-1, main_embeddings.shape[1])
    if combined_labels.ndim == 1:
        combined_labels = combined_labels.reshape(-1, main_labels.shape[1])

    # Concatenate with the original embeddings and labels
    final_embeddings = np.concatenate([main_embeddings, combined_embeddings], axis=0)
    final_labels = np.concatenate([main_labels, combined_labels], axis=0)

    return final_embeddings, final_labels


# Function to summarize label distribution
def summarize_labels(labels, save_path):
    label_sums = labels.sum(axis=0)
    label_summary_df = pd.DataFrame(label_sums, columns=['Count'])
    label_summary_df.index.name = 'Class'
    
    # Save the summary
    label_summary_path = os.path.join(save_path, 'label_summary.csv')
    label_summary_df.to_csv(label_summary_path)
    print(f"Label summary saved to {label_summary_path}")
    
    return label_summary_df


def collate_results(result, collated_results_path):
    # Check if the collated results file exists
    if not os.path.exists(collated_results_path):
        collated_df = pd.DataFrame()
    else:
        collated_df = pd.read_csv(collated_results_path)
    
    # Convert the result to a DataFrame and concatenate
    result_df = pd.DataFrame([result])
    collated_df = pd.concat([collated_df, result_df], ignore_index=True)
    
    # Save the collated results
    collated_df.to_csv(collated_results_path, index=False)


def train_and_evaluate_model(embeddings, labels, save_dir, n_jobs=-1, aug_percent=None, data_percent=None, collated_results_path=None):
    # Apply PCA with joblib parallelization
    pca = PCA(n_components=100)
    features_reduced = pca.fit_transform(embeddings)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features_reduced, labels, test_size=0.2, random_state=42)
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    param_grid = {
        'estimator__n_estimators': [300],
        'estimator__max_depth': [50]
    }

    # Train a classifier with parallel processing
    clf = MultiOutputClassifier(RandomForestClassifier(random_state=42, n_jobs=n_jobs, n_estimators=300, max_depth=50),n_jobs=n_jobs)
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_micro', n_jobs=n_jobs, error_score='raise')
    start_train = time.time()
    grid_search.fit(X_train, y_train)
    end_train = time.time()
    print(f"Training completed in {end_train - start_train:.2f} seconds")

    # Predictions and scoring
    start_pred = time.time()
    y_pred = grid_search.predict(X_test)
    end_pred = time.time()
    print(f"Prediction completed in {end_pred - start_pred:.2f} seconds")
    
    precision = precision_score(y_test, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"Overall Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}, F1 Weighted: {f1_weighted:.4f}")
    
    result = {
        'Model': "ViT-bigG-14-CLIPA-336_datacomp1b",
        'PCA_Components': 70,
        'Best_Params': grid_search.best_params_,
        'Precision': precision,
        'Recall': recall,
        'F1_Score_Micro': f1_micro,
        'F1_Score_Macro': f1_macro,
        'F1_Score_Weighted': f1_weighted,
        'Train_Time': end_train - start_train,
        'Predict_Time': end_pred - start_pred,
        'Amount of Augmentation': aug_percent,
        'Amount of Original Data': data_percent
    }
    results_df = pd.DataFrame([result])
    results_df.to_csv(os.path.join(save_dir, 'overall_results.csv'), index=False)
    
    dump(grid_search.best_estimator_, os.path.join(save_dir, 'model.joblib'))
    
    # Collate results
    if collated_results_path:
        collate_results(result, collated_results_path)

    # Class-level metrics
    class_report = classification_report(y_test, y_pred, target_names=[f'Class_{i}' for i in range(y_test.shape[1])], output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df.to_csv(os.path.join(save_dir, 'class_level_results.csv'), index=True)
    print(f"Model, overall results, and class-level results saved to {save_dir}.")

    return precision, recall, f1_micro, f1_macro, f1_weighted


# Function to plot the metrics
def plot_metrics(metrics, title, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    for metric_name, values in metrics.items():
        plt.plot(values['x'], values['y'], label=metric_name)
    plt.xlabel('Percentage of Original Data')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# Define a function to run baseline model
def run_baseline_model(main_embeddings, main_labels, save_dir_base, n_jobs=-1, collated_results_path=None):
    summarize_data(main_embeddings, main_labels, data_type="Original")
    save_dir_baseline = os.path.join(save_dir_base, 'baseline_model')
    os.makedirs(save_dir_baseline, exist_ok=True)
    train_and_evaluate_model(main_embeddings, main_labels, save_dir_baseline, n_jobs=n_jobs, aug_percent=None, data_percent=None, collated_results_path=collated_results_path)
    
    # Summarize the labels for the baseline model
    summarize_labels(main_labels, save_dir_baseline)

def run_incremental_augmentation(main_embeddings, main_labels, augmentation_embeddings, augmentation_labels, save_dir_base, data_percentages, n_jobs=-1, collated_results_path=None):
    metrics_no_aug = {'F1 f1_micro': {'x': [], 'y': []}, 
                      'F1 f1_weighted': {'x': [], 'y': []}, 
                      'F1 f1_macro': {'x': [], 'y': []}, 
                      'Precision': {'x': [], 'y': []}, 
                      'Recall': {'x': [], 'y': []}}

    metrics_with_aug = {'F1 f1_micro': {'x': [], 'y': []}, 
                        'F1 f1_weighted': {'x': [], 'y': []}, 
                        'F1 f1_macro': {'x': [], 'y': []}, 
                        'Precision': {'x': [], 'y': []}, 
                        'Recall': {'x': [], 'y': []}}

    for percentage in data_percentages:
        if percentage == 100:
            reduced_embeddings = main_embeddings
            reduced_labels = main_labels
        else:
            train_size_fraction = percentage / 100.0
            reduced_embeddings, _, reduced_labels, _ = train_test_split(main_embeddings, main_labels, train_size=train_size_fraction, random_state=42)
      
        # Train and evaluate the model without augmentation
        save_dir_no_aug = os.path.join(save_dir_base, f"{percentage}_percent_data_no_aug")
        os.makedirs(save_dir_no_aug, exist_ok=True)
        precision, recall, f1_micro, f1_macro, f1_weighted = train_and_evaluate_model(
            reduced_embeddings, reduced_labels, save_dir_no_aug, 
            n_jobs=n_jobs, aug_percent=None, data_percent=percentage, collated_results_path=collated_results_path
        )
        metrics_no_aug['F1 f1_weighted']['x'].append(percentage)
        metrics_no_aug['F1 f1_weighted']['y'].append(f1_weighted)
        metrics_no_aug['F1 f1_macro']['x'].append(percentage)
        metrics_no_aug['F1 f1_macro']['y'].append(f1_macro)
        metrics_no_aug['F1 f1_micro']['x'].append(percentage)
        metrics_no_aug['F1 f1_micro']['y'].append(f1_micro)
        metrics_no_aug['Precision']['x'].append(percentage)
        metrics_no_aug['Precision']['y'].append(precision)
        metrics_no_aug['Recall']['x'].append(percentage)
        metrics_no_aug['Recall']['y'].append(recall)
        
        if augmentation_embeddings is not None and augmentation_labels is not None:
            for aug_fraction in [0.05, 0.25, 0.5, 0.75, 1.0]:  # Incremental steps of augmentation data
                if aug_fraction == 1.0:
                    # Combine all augmented data with all reduced original data
                    combined_embeddings, combined_labels = combine_embeddings(
                        reduced_embeddings, reduced_labels,
                        augmentation_embeddings, augmentation_labels
                    )
                else:
                    # Normal sampling process for other fractions
                    num_aug_samples = int(len(augmentation_embeddings) * aug_fraction)
                    if num_aug_samples > 0 and num_aug_samples <= len(augmentation_embeddings):
                        aug_embeddings_sampled, _, aug_labels_sampled, _ = train_test_split(
                            augmentation_embeddings, augmentation_labels, 
                            train_size=aug_fraction, 
                            random_state=42
                        )
                    else:
                        print("else triggered")
                        aug_embeddings_sampled = np.array([]).reshape(0, augmentation_embeddings.shape[1])
                        aug_labels_sampled = np.array([]).reshape(0, augmentation_labels.shape[1])
                    
                    # Ensure correct size and avoid duplicates
                    combined_embeddings, combined_labels = combine_embeddings(
                        reduced_embeddings, reduced_labels, 
                        aug_embeddings_sampled, 
                        aug_labels_sampled
                    )
        
                save_dir_with_aug = os.path.join(save_dir_base, f"{percentage}_percent_data_with_aug_{int(aug_fraction*100)}_percent")
                os.makedirs(save_dir_with_aug, exist_ok=True)
                
                # Train and evaluate the model
                precision, recall, f1_micro, f1_macro, f1_weighted = train_and_evaluate_model(
                    combined_embeddings, combined_labels, save_dir_with_aug, 
                    n_jobs=n_jobs, aug_percent=aug_fraction*100, data_percent=percentage, collated_results_path=collated_results_path
                )
        
                # Store the results for plotting
                metrics_with_aug['F1 f1_weighted']['x'].append(f"{percentage}+{int(aug_fraction*100)}%")
                metrics_with_aug['F1 f1_weighted']['y'].append(f1_weighted)
                metrics_with_aug['F1 f1_macro']['x'].append(f"{percentage}+{int(aug_fraction*100)}%")
                metrics_with_aug['F1 f1_macro']['y'].append(f1_macro)
                metrics_with_aug['F1 f1_micro']['x'].append(f"{percentage}+{int(aug_fraction*100)}%")
                metrics_with_aug['F1 f1_micro']['y'].append(f1_micro)
                metrics_with_aug['Precision']['x'].append(f"{percentage}+{int(aug_fraction*100)}%")
                metrics_with_aug['Precision']['y'].append(precision)
                metrics_with_aug['Recall']['x'].append(f"{percentage}+{int(aug_fraction*100)}%")
                metrics_with_aug['Recall']['y'].append(recall)
        
                # Summarize the labels after augmentation
                summarize_labels(combined_labels, save_dir_with_aug)
    
    
    # Plot and save the metrics
    plot_metrics(metrics_no_aug, "Model Performance Without Data Augmentation", "Score", os.path.join(save_dir_base, 'performance_no_aug.png'))
    plot_metrics(metrics_with_aug, "Model Performance With Incremental Data Augmentation", "Score", os.path.join(save_dir_base, 'performance_with_aug.png'))


# Load the main embeddings and labels
main_embeddings_path = 'results/clip_embeddings.npy'
main_labels_path = 'results/clip_labels.npy'

if not os.path.exists(main_embeddings_path) or not os.path.exists(main_labels_path):
    raise FileNotFoundError(f"Main embeddings or labels not found. Check the paths:\n{main_embeddings_path}\n{main_labels_path}")

main_embeddings = np.load(main_embeddings_path)
main_labels = np.load(main_labels_path)
name_list =  [
    "Droplets_only", 
    "Impurities_only", 
    "regular_crystal_only", 
    "Bubbles_only",
    "impurites_regular",
    'impurites_bubbles',
    'droplets_and_impurites'
]
# Load the augmented embeddings and labels from the specific permutation
try:
    augmentation_embeddings_path = load_and_save_permutations(
        name_list, 
        "embeddings"
    )
    augmentation_labels_path = load_and_save_permutations(
        name_list, 
        "labels"
    )
except FileNotFoundError as e:
    print(e)
    augmentation_embeddings_path = None
    augmentation_labels_path = None

if augmentation_embeddings_path and augmentation_labels_path:
    augmentation_embeddings = np.load(augmentation_embeddings_path)
    augmentation_labels = np.load(augmentation_labels_path)
else:
    augmentation_embeddings = None
    augmentation_labels = None

# Run baseline model on full dataset with parallel processing
run_baseline_model(main_embeddings, main_labels, save_dir_base="threshold_testing_results", n_jobs=-1, collated_results_path="collated_results.csv")

# Run incremental augmentation testing with parallel processing
run_incremental_augmentation(
    main_embeddings, main_labels,
    augmentation_embeddings, 
    augmentation_labels,
    save_dir_base="threshold_testing_results", 
    data_percentages=[5, 10, 25, 50, 75, 100],  # Adjust this list based on the required percentages
    n_jobs=-1,  # Use all available cores for parallel processing
    collated_results_path="collated_results.csv"
)


