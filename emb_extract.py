
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from open_clip import create_model_and_transforms
import time

# Load the data
data_df = pd.read_csv("/opt/data/AnSh/backup/master_key.csv")

# Function to get image label
def get_image_label(row):
    img_path = os.path.join("/opt/data/AnSh/cropped_df_data", row['File_Path'].replace('/', os.sep))
    label = row[['Object present', 'Needle-like crystal', 'Elongated crystal', 'Platelet crystal',
                 'Regular crystal', 'Impurity', 'Agglomerated crystals', 'Bubbles', 'Droplets',
                 'Too concentrated']].values.astype(int)
    return img_path, label

# Function to get CLIP embeddings for a batch of images
def get_clip_embeddings_batch(image_paths, preprocess, model, device):
    try:
        images = [preprocess(Image.open(image_path)).unsqueeze(0).to(device) for image_path in image_paths]
        images = torch.cat(images, dim=0)
        with torch.no_grad():
            image_features = model.encode_image(images)
        return image_features.cpu().numpy()
    except Exception as e:
        print(f"Error processing images {image_paths}: {e}")
        return None

# Function to process a batch of rows
def process_rows(rows, preprocess, model, device):
    img_paths = []
    labels = []
    for _, row in rows.iterrows():
        img_path, label = get_image_label(row)
        if os.path.exists(img_path):
            img_paths.append(img_path)
            labels.append(label)

    if not img_paths:
        return None, None

    embeddings = get_clip_embeddings_batch(img_paths, preprocess, model, device)
    if embeddings is not None:
        return embeddings, label
    return None, None


def process_row(row):
    img_path, label = get_image_label(row)
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return None, None
    embeddings = get_clip_embeddings_batch(img_paths, preprocess, model, device)
    if embedding is not None:
        return embedding, label
    return None, None

# Multi-threaded processing
def process_data(df, preprocess, model, device, batch_size=8, max_workers=50):
    features = []
    labels = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch_rows = df.iloc[i:i+batch_size]
            futures.append(executor.submit(process_rows, batch_rows, preprocess, model, device))
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                batch_features, batch_labels = result
                features.extend(batch_features)
                labels.extend(batch_labels)
    
    return np.array(features), np.array(labels)

# Function to run the processing for each model using multiple GPUs and running multiple models at once
def run_for_models(models_df, data_df, test_mode=False, sample_size=100, batch_size=8, models_per_run=2):
    if test_mode:
        data_df = data_df.sample(n=sample_size, random_state=42)
        print(f"Running in test mode with a sample size of {sample_size}")

    devices = ["cuda:0", "cuda:1"] if torch.cuda.device_count() > 1 else ["cuda"]
    
    model_groups = [models_df.iloc[i:i+models_per_run] for i in range(0, len(models_df), models_per_run)]
    
    for model_group in model_groups:
        group_start_time = time.time()
        futures = []
        model_times = {}
        
        with ThreadPoolExecutor(max_workers=len(model_group)) as executor:
            for idx, model_row in model_group.iterrows():
                model_name = model_row['name']
                pretrained = model_row['pretrained']
                
                device = devices[idx % len(devices)]
                print(f"Loading model {model_name} with pretrained {pretrained} on {device}")
                
                model, _, preprocess = create_model_and_transforms(model_name, pretrained, device)
                model.to(device)
                
                model_start_time = time.time()
                
                # Each model processes the entire dataset (or a sample if in test mode)
                future = executor.submit(process_data, data_df, preprocess, model, device, batch_size)
                futures.append((future, model_name, model_start_time))
            
            features_list = []
            labels_list = []
            
            for future, model_name, model_start_time in futures:
                features, labels = future.result()
                model_end_time = time.time()
                model_times[model_name] = model_end_time - model_start_time

                if features is not None:
                    features_list.append(features)
                    print(f"Features shape: {features.shape}")
                if labels is not None:
                    labels_list.append(labels)
                    print(f"Labels shape: {labels.shape}")
        
        for features, labels, model_row in zip(features_list, labels_list, model_group.iterrows()):
            _, model_row = model_row
            model_name = model_row['name']
            pretrained = model_row['pretrained']
            
            print(f"Saving results for model {model_name} with pretrained {pretrained}")
            np.save(f"{model_name}_{pretrained}_clip_combined_embeddings.npy", features)
            np.save(f"{model_name}_{pretrained}_clip_labels.npy", labels)
            
            print(f"Embeddings and labels for model {model_name} with pretrained {pretrained} have been saved.")
        
        group_end_time = time.time()
        group_time = group_end_time - group_start_time
        print(f"Time taken for processing model group: {group_time} seconds")
        
        for model_name, model_time in model_times.items():
            print(f"Time taken for processing model {model_name}: {model_time} seconds")
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Models DataFrame
    # Models DataFrame
models_df = pd.DataFrame({
    'name': [
        'ViT-bigG-14-CLIPA-336', 'ViT-SO400M-14-SigLIP-384', 'EVA02-E-14-plus',
        'ViT-H-14-quickgelu', 'ViT-H-14-378-quickgelu',
        'RN50', 'RN50-quickgelu', 'RN101', 'RN101-quickgelu',
        'RN50', 'RN50-quickgelu', 'RN101', 'RN101-quickgelu'
    ],
    'pretrained': [
        'datacomp1b', 'webli', 'laion2b_s9b_b144k', 
        'dfn5b', 'dfn5b',
        'openai', 'openai', 'openai', 'openai',
        'yfcc15m', 'cc12m', 'yfcc15m', 'openai'
    ]
})
# Run the processing for each model
run_for_models(models_df, data_df, test_mode=False, sample_size=100, batch_size=32)
