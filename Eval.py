import numpy as np
import os
import pandas as pd
from joblib import load
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Directory containing embedding files and models
embedding_dir = "Embeddings"

# Read the model results CSV to get model names and PCA components
model_results_path = 'Embeddings/model_results.csv'
object_level_scores_path = 'Embeddings/object_level_scores.csv'

model_results_df = pd.read_csv(model_results_path)
object_level_scores_df = pd.read_csv(object_level_scores_path)

# Get the best PCA configuration for each model
best_pca_per_model = model_results_df.loc[model_results_df.groupby('Model')['Best_Score'].idxmax()]

# Define the mapping of labels to their actual names
label_to_name = {
    0: 'Object present',
    1: 'Needle-like crystal',
    2: 'Elongated crystal',
    3: 'Platelet crystal',
    4: 'Regular crystal',
    5: 'Impurity',
    6: 'Agglomerated crystals',
    7: 'Bubbles',
    8: 'Droplets',
    9: 'Too concentrated'
}

# Function to get actual label names for combinations
def get_label_names(comb):
    return [label_to_name[int(col.split('_')[-1])] for col in comb]

# Function to evaluate models and generate summaries
def evaluate_models(model_df):
    eval_results = []
    precision_recall_curves = []

    for _, row in model_df.iterrows():
        clip_model_name = row['Model']
        pca_components = row['PCA_Components']
        
        label_file = f"{clip_model_name}_clip_labels.npy"
        embedding_file = f"{clip_model_name}_clip_combined_embeddings.npy"  # Adjusted file name
        model_file = f"{clip_model_name}_model_pca_{pca_components}.joblib"
        
        label_path = os.path.join(embedding_dir, label_file)
        embedding_path = os.path.join(embedding_dir, embedding_file)
        model_path = os.path.join(embedding_dir, model_file)
        
        if not os.path.exists(label_path) or not os.path.exists(embedding_path) or not os.path.exists(model_path):
            print(f"Required file not found for {clip_model_name} with PCA {pca_components}, skipping.")
            continue
        
        # Load the model, features, and labels
        model = load(model_path)
        features = np.load(embedding_path)
        labels = np.load(label_path)
        
        pca = PCA(n_components=pca_components)
        features_reduced = pca.fit_transform(features[:, :-10])
        
        X_train, X_test, y_train, y_test = train_test_split(features_reduced, labels, test_size=0.2, random_state=42)
        
        y_pred = model.predict(X_test)
        
        if y_test.shape != y_pred.shape:
            print(f"Mismatch in shapes: y_test {y_test.shape}, y_pred {y_pred.shape}, skipping.")
            continue
        
        precision = precision_score(y_test, y_pred, average='micro', zero_division=1)
        recall = recall_score(y_test, y_pred, average='micro', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='micro', zero_division=1)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
        
        eval_result = {
            'Model': clip_model_name,
            'PCA_Components': pca_components,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Macro_F1_Score': macro_f1,
            'Weighted_F1_Score': weighted_f1,
            'Train_Time': row['Train_Time'],
            'Predict_Time': row['Predict_Time']
        }
        
        # Calculate per-object precision, recall, and F1 scores
        num_labels = y_test.shape[1]
        for i in range(num_labels):
            eval_result[f'Precision_Label_{i}'] = precision_score(y_test[:, i], y_pred[:, i], zero_division=1)
            eval_result[f'Recall_Label_{i}'] = recall_score(y_test[:, i], y_pred[:, i], zero_division=1)
            eval_result[f'F1_Score_Label_{i}'] = f1_score(y_test[:, i], y_pred[:, i], zero_division=1)
        
        eval_results.append(eval_result)
        
        # Collect data for precision-recall curve
        y_scores = np.array([model.estimators_[i].predict_proba(X_test)[:, 1] for i in range(num_labels)]).T
        for label in range(num_labels):
            precision, recall, _ = precision_recall_curve(y_test[:, label], y_scores[:, label])
            precision_recall_curves.append({
                'Model': clip_model_name,
                'PCA_Components': pca_components,
                'Label': label,
                'Precision': precision,
                'Recall': recall
            })

    # Convert evaluation results to a DataFrame
    eval_results_df = pd.DataFrame(eval_results)

    return eval_results_df, precision_recall_curves

# Evaluate models based on the best PCA configuration
best_pca_eval_results_df, precision_recall_curves = evaluate_models(best_pca_per_model)

# Save the evaluation results to a CSV file
best_pca_eval_results_df.to_csv('best_pca_evaluation_results.csv', index=False)

# Identify worst-performing combinations of classes at the prediction level
def find_worst_combinations(df, num_combinations=5):
    label_columns = [col for col in df.columns if 'F1_Score_Label_' in col]
    worst_combinations = []
    
    for i in range(2, len(label_columns) + 1):
        combs = combinations(label_columns, i)
        for comb in combs:
            avg_f1 = df[list(comb)].mean(axis=1).mean()
            worst_combinations.append((comb, avg_f1))
    
    worst_combinations = sorted(worst_combinations, key=lambda x: x[1])[:num_combinations]
    return worst_combinations

# Find rows where both labels in a combination are present
def filter_rows_for_combination(labels, comb):
    indices = np.where(np.all(labels[:, [int(col.split('_')[-1]) for col in comb]] == 1, axis=1))[0]
    return indices

worst_combinations = find_worst_combinations(best_pca_eval_results_df)
print("Worst-performing combinations of classes:")
for comb, avg_f1 in worst_combinations:
    print(f"Combination: {get_label_names(comb)}, Average F1 Score: {avg_f1}")

# Visualize the worst-performing combinations
for comb, avg_f1 in worst_combinations:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=best_pca_eval_results_df[list(comb)])
    plt.title(f'Worst-performing combination: {get_label_names(comb)}')
    plt.xlabel('Labels')
    plt.ylabel('F1 Score')
    plt.show()

# Generate precision-recall curves for top models
top_models = best_pca_eval_results_df.sort_values(by='F1_Score', ascending=False)['Model'].unique()[:10]
for model_name in top_models:
    model_curves = [curve for curve in precision_recall_curves if curve['Model'] == model_name]
    plt.figure(figsize=(10, 6))
    for curve in model_curves:
        plt.plot(curve['Recall'], curve['Precision'], marker='.', label=f'Label {label_to_name[curve["Label"]]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend()
    plt.show()

# Additional Analysis
# Training Time vs. F1 Score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=best_pca_eval_results_df, x='Train_Time', y='F1_Score', hue='Model', style='Model', markers=True)
plt.title('Training Time vs. F1 Score')
plt.xlabel('Training Time (seconds)')
plt.ylabel('F1 Score')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
