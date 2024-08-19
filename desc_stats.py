import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


data_old = pd.read_csv("/opt/data/AnSh/backup/master_key.csv")


row_sum = data_old.iloc[:, 12:].sum()
row_mean = data_old.iloc[:, 12:].mean()
[row_sum, row_mean]

for i in range(0,len(data_old.columns)):
    if len(pd.unique(data_old.iloc[:,i])) <= 12:
        print([data_old.columns[i],pd.unique(data_old.iloc[:,i])])

data_old.describe()




# Filter images based on labels and temperature
def filter_images_by_label_and_temperature(data_df, label_name, temperature_threshold, combination=False):
    # Columns related to crystal labels and temperature
    crystal_columns = ['Needle-like crystal', 'Elongated crystal', 'Platelet crystal', 
                       'Regular crystal', 'Impurity', 'Agglomerated crystals', 
                       'Bubbles', 'Droplets', 'Too concentrated', 'Temperatures']
    
    filtered_df = data_df[crystal_columns + ['File_Path']]  # Keep File_Path for final selection
    
    if combination:
        # Find images with the crystal type and the most common co-occurring crystal, filtered by temperature
        filtered_df = filtered_df[(filtered_df[label_name] == 1) & (filtered_df['Temperatures'] <= temperature_threshold)]
    else:
        # Find images with only the crystal type and filtered by temperature
        filtered_df = filtered_df[
            (filtered_df[label_name] == 1) & 
            (filtered_df[crystal_columns[:-1]].sum(axis=1) == 1) &  # Ensure only one crystal type is present
            (filtered_df['Temperatures'] <= temperature_threshold)  # Apply temperature filter
        ]
    
    return filtered_df

# Select images for the grid
def select_images_for_grid(data_df, temperature_threshold):
    labels = ['Needle-like crystal', 'Elongated crystal', 'Platelet crystal', 
              'Regular crystal', 'Impurity', 'Agglomerated crystals', 
              'Bubbles', 'Droplets', 'Too concentrated']
    
    selected_images = []
    for label in labels:
        filtered_df = filter_images_by_label_and_temperature(data_df, label, temperature_threshold)
        if len(filtered_df) == 0:  # If no isolated images found, look for combinations
            filtered_df = filter_images_by_label_and_temperature(data_df, label, temperature_threshold, combination=True)
        if len(filtered_df) > 0:
            filtered_df = filtered_df.sample(frac=1, random_state=None).reset_index(drop=True)  # Shuffle the filtered DataFrame
            selected_images.append(filtered_df.iloc[0]['File_Path'])  # Select the first matching image path

    selected_images_df = data_df[data_df['File_Path'].isin(selected_images)]  # Retrieve full rows for selected images
    return selected_images_df[:9]  # Select only the first 9 unique images

# Create and display the grid
def create_image_grid(selected_images):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i, ax in enumerate(axes.flat):
        img_path = os.path.join("/opt/data/AnSh/cropped_df_data", selected_images.iloc[i]['File_Path'].replace('/', os.sep))
        label_text = ', '.join([selected_images.columns[j] for j in range(len(selected_images.columns)) if selected_images.iloc[i, j] == 1 and selected_images.columns[j] in ['Needle-like crystal', 'Elongated crystal', 'Platelet crystal', 'Regular crystal', 'Impurity', 'Agglomerated crystals', 'Bubbles', 'Droplets', 'Too concentrated']])
        
        # Debugging output
        print(f"Loading image: {img_path}")
        print(f"Labels: {label_text}")
        
        if not os.path.exists(img_path):
            print(f"Image does not exist: {img_path}")  # Debugging statement
            continue

        img = Image.open(img_path).convert('RGB')  # Convert image to RGB
        ax.imshow(img)
        ax.set_title(label_text)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Run the selection and grid creation process
data_df = pd.read_csv("/opt/data/AnSh/backup/master_key.csv")
temperature_threshold = 36  # Set your desired temperature threshold here
selected_images = select_images_for_grid(data_df, temperature_threshold)

# Check if any images were selected
if selected_images.empty:
    print("No images found with the specified criteria.")
else:
    create_image_grid(selected_images)

