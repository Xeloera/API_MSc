
import numpy as np
import itertools
import os
import numpy as np
import os
import time
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import itertools

# List of names to create permutations, including additional combinations
name_list = [
    "Droplets_only", 
    "Impurities_only", 
    "regular_crystal_only", 
    "Bubbles_only"
]

# Function to load, concatenate, and save arrays for given names
def load_and_save_permutations(names, array_type):
    main_array = np.load(f'results/clip_{array_type}.npy')
    arrays = [np.load(f'results/clip_{array_type}_{name}.npy') for name in names]
    merged_array = np.concatenate([main_array] + arrays, axis=0)
    filename = '_and_'.join(names)
    np.save(f'results_final/{array_type}/clip_{array_type}_{filename}_final.npy', merged_array)
    print(f"Merged array saved to 'results_final/{array_type}/clip_{array_type}_{filename}_final.npy'")

# Ensure directories exist
os.makedirs('results_final/embeddings', exist_ok=True)
os.makedirs('results_final/labels', exist_ok=True)

# List to store all combinations
all_combinations = []

# Generate all unique combinations from 1 element to the total number of elements in name_list
for r in range(1, len(name_list) + 1):
    combinations = itertools.combinations(name_list, r)
    for combination in combinations:
        all_combinations.append(combination)
        load_and_save_permutations(combination, "embeddings")
        load_and_save_permutations(combination, "labels")

# Calculate the number of combinations
num_combinations = len(all_combinations)

# Print out the list of combinations and the total number
for combo in all_combinations:
    print(combo)

print(f"Total number of combinations: {num_combinations}")




# List of names to create permutations, including additional combinations
name_list = [
    "Droplets_only", 
    "Impurities_only", 
    "regular_crystal_only", 
    "Bubbles_only",
    "impurites_regular",
    'impurites_bubbles',
    'droplets_and_impurites'
]

# Function to load, concatenate, and save arrays for given names
def load_and_save_permutations(names, array_type):
    main_array = np.load(f'results/clip_{array_type}.npy')
    arrays = [np.load(f'results/clip_{array_type}_{name}.npy') for name in names]
    merged_array = np.concatenate([main_array] + arrays, axis=0)
    filename = '_and_'.join(names)
    np.save(f'results_final/{array_type}/clip_{array_type}_{filename}_final.npy', merged_array)
    print(f"Merged array saved to 'results_final/{array_type}/clip_{array_type}_{filename}_final.npy'")

# Ensure directories exist
os.makedirs('results_final/embeddings', exist_ok=True)
os.makedirs('results_final/labels', exist_ok=True)
os.makedirs('results_final/finished_models', exist_ok=True)
'''
# List to store all combinations
all_combinations = []

# Generate all unique combinations from 1 element to the total number of elements in name_list
for r in range(1, len(name_list) + 1):
    combinations = itertools.combinations(name_list, r)
    for combination in combinations:
        all_combinations.append(combination)
        load_and_save_permutations(combination, "embeddings")
        load_and_save_permutations(combination, "labels")

# Calculate the number of combinations
num_combinations = len(all_combinations)

# Print out the list of combinations and the total number
for combo in all_combinations:
    print(combo)

print(f"Total number of combinations: {num_combinations}")
'''
# Function to train and evaluate the model
def train_and_evaluate_model(name):
    labels = np.load(f"results_final/labels/clip_labels_{name}_final.npy")
    embeddings = np.load(f"results_final/embeddings/clip_embeddings_{name}_final.npy")

    pca = PCA(n_components=70)
    features_reduced = pca.fit_transform(embeddings)
    X_train, X_test, y_train, y_test = train_test_split(features_reduced, labels, test_size=0.2, random_state=42)

    clf = MultiOutputClassifier(RandomForestClassifier(random_state=42, n_jobs=-1), n_jobs=-1)
    param_grid = {
        'estimator__n_estimators': [200],
        'estimator__max_depth': [50]
    }

    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_micro', n_jobs=-1, error_score='raise')

    start_train = time.time()
    grid_search.fit(X_train, y_train)
    end_train = time.time()

    # Predictions and scoring
    start_pred = time.time()
    y_pred = grid_search.predict(X_test)
    end_pred = time.time()

    precision = precision_score(y_test, y_pred, average='micro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='micro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='micro', zero_division=1)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

    result = {
        'Combination': name,
        'Model': "ViT-bigG-14-CLIPA-336_datacomp1b",
        'PCA_Components': 70,
        'Best_Params': grid_search.best_params_,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'Macro_F1_Score': macro_f1,
        'Weighted_F1_Score': weighted_f1,
        'Train_Time': end_train - start_train,
        'Predict_Time': end_pred - start_pred
    }

    # Save results to DataFrame and CSV
    results_df = pd.DataFrame([result])
    results_df.to_csv(f"results_final/{name}_results.csv", index=False)
    
    # Save model
    dump(grid_search.best_estimator_, f"results_final/finished_models/{name}_final.joblib")

    # Save evaluation results separately
    eval_results = []

    eval_result = {
        'Combination': name,
        'Model': "ViT-bigG-14-CLIPA-336_datacomp1b",
        'PCA_Components': 70,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'Macro_F1_Score': macro_f1,
        'Weighted_F1_Score': weighted_f1
    }
    
    # Calculate per-object precision, recall, and F1 scores
    num_labels = y_test.shape[1]
    for i in range(num_labels):
        eval_result[f'Precision_Label_{i}'] = precision_score(y_test[:, i], y_pred[:, i], zero_division=1)
        eval_result[f'Recall_Label_{i}'] = recall_score(y_test[:, i], y_pred[:, i], zero_division=1)
        eval_result[f'F1_Score_Label_{i}'] = f1_score(y_test[:, i], y_pred[:, i], zero_division=1)

    eval_results.append(eval_result)

    eval_results_df = pd.DataFrame(eval_results)
    eval_results_df.to_csv(f'results_final/{name}_object_level_scores.csv', index=False)

    print(f"Results for {name} saved.")

# Train and evaluate model for each combination

all_combinations = [('Droplets_only', 'Impurities_only', 'regular_crystal_only', 'Bubbles_only', 'impurites_regular', 'impurites_bubbles', 'droplets_and_impurites'),
                    ('Droplets_only', 'Impurities_only', 'regular_crystal_only', 'Bubbles_only')]
for combo in all_combinations:
    name = '_and_'.join(combo)
    train_and_evaluate_model(name)

# Collect all results CSV files into one large table



all_results = []
for csv_file in os.listdir('results_final'):
    if csv_file.endswith('_results.csv'):
        df = pd.read_csv(os.path.join('results_final', csv_file))
        all_results.append(df)

combined_results_df = pd.concat(all_results, ignore_index=True)
combined_results_df.to_csv('results_final/combined_results.csv', index=False)

print("All results combined and saved to 'results_final/combined_results.csv'")
