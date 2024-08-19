import os
import shutil
import json
import pandas as pd
# Function to generate prompts from the dataset
def generate_prompt(row):
    solute = row['Solute']
    solvent = row['Solvent']
    crystals = []
    crystal_columns = ['Needle-like crystal', 'Elongated crystal', 'Platelet crystal', 'Regular crystal', 'Agglomerated crystals']
    for col in crystal_columns:
        if row[col] == 1:
            crystals.append(col)
    impurities = "impurities present" if row['Impurity'] != 0 else "no impurities present"
    additional_conditions = []
    additional_columns = ['Bubbles', 'Droplets', 'Too concentrated']
    for col in additional_columns:
        if row[col] == 1:
            additional_conditions.append(col)
    crystal_desc = ', '.join(crystals) if crystals else "no specific type"
    conditions_desc = ', '.join(additional_conditions) if additional_conditions else "none"
    prompt = (
        f"A crystal formation consisting of {crystal_desc} crystals with solute {solute} in solvent {solvent}, "
        f"{impurities}. "
        f"Additional conditions: {conditions_desc}."
    )
    return prompt


def prepare_dataset(dataframe, base_path, dataset_dir):
    images_dir = os.path.join(dataset_dir, "images")
    metadata_path = os.path.join(dataset_dir, "metadata.jsonl")

    # Create necessary directories
    os.makedirs(images_dir, exist_ok=True)

    # Create the metadata.jsonl file
    with open(metadata_path, 'w') as f:
        for _, row in dataframe.iterrows():
            image_path = os.path.join(base_path, row['File_Path'].replace('/', os.sep))
            prompt = generate_prompt(row)
            img_filename = os.path.basename(image_path)
            new_img_path = os.path.join(images_dir, img_filename)

            # Copy image to the images directory
            if not os.path.exists(new_img_path):
                shutil.copy(image_path, new_img_path) 

            # Write the metadata entry
            metadata = {
                "file_name": f"images/{img_filename}",
                "text": prompt
            }
            f.write(json.dumps(metadata) + "\n")

# Read the master key CSV
data = pd.read_csv("master_key.csv")

# Define base paths

base_path = "/opt/data/AnSh/cropped_df_data"

# Define the dataset directory
dataset_dir = "diffusers/examples/text_to_image/dataset"
prepare_dataset(data, base_path, dataset_dir)
