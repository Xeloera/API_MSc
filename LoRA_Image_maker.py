import numpy as np
import os
import pandas as pd
from PIL import Image, ImageOps
from diffusers import DiffusionPipeline
import torch
from diffusers import (EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, 
                       DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,
                       HeunDiscreteScheduler, LMSDiscreteScheduler, DEISMultistepScheduler, UniPCMultistepScheduler)
import open_clip
from diffusers import AutoPipelineForText2Image
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from accelerate import PartialState
# Function to initialize different schedulers

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

# Define prompts for generating images based on class labels
prompts = {
    "Object present": "A microscopic image of a formation of objects",
    "Needle-like crystal": "A microscopic image of a formation of Needle-like crystals",
    "Elongated crystal": "A microscopic image of a formation of Elongated crystals",
    "Platelet crystal": "A microscopic image of a formation of Platelet crystals",
    "Regular crystal": "A microscopic image of formation of Regular crystals",
    "Impurity": "A microscopic image of formation of impurities",
    "Agglomerated crystals": "A microscopic image of a formation of Agglomerated crystals",
    "Bubbles": "A microscopic image of bubbles",
    "Droplets": "A microscopic image of droplets",
    "Too concentrated": "A microscopic image of a too concentrated solution"
}

# Load the data
data_df = pd.read_csv("/opt/data/AnSh/backup/master_key.csv")


def initialize_scheduler(scheduler_type):
    if scheduler_type == 'Euler':
        return EulerDiscreteScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
    elif scheduler_type == 'Euler a':
        return EulerAncestralDiscreteScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
    elif scheduler_type == 'DPM Solver':
        return DPMSolverMultistepScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000, use_karras_sigmas=False)
    elif scheduler_type == 'DPM Solver Karras':
        return DPMSolverMultistepScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000, use_karras_sigmas=True)
    elif scheduler_type == 'DPM Solver SDE':
        return DPMSolverMultistepScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000, algorithm_type="sde-dpmsolver++")
    elif scheduler_type == 'DPM Solver SDE Karras':
        return DPMSolverMultistepScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif scheduler_type == 'DPM++ 2S a':
        return DPMSolverSinglestepScheduler()
    elif scheduler_type == 'DPM++ 2S a Karras':
        return DPMSolverSinglestepScheduler(use_karras_sigmas=True)
    elif scheduler_type == 'DPM++ SDE':
        return DPMSolverSinglestepScheduler()
    elif scheduler_type == 'DPM++ SDE Karras':
        return DPMSolverSinglestepScheduler(use_karras_sigmas=True)
    elif scheduler_type == 'DPM2':
        return KDPM2DiscreteScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
    elif scheduler_type == 'DPM2 Karras':
        return KDPM2DiscreteScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000, use_karras_sigmas=True)
    elif scheduler_type == 'DPM2 a':
        return KDPM2AncestralDiscreteScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
    elif scheduler_type == 'DPM2 a Karras':
        return KDPM2AncestralDiscreteScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000, use_karras_sigmas=True)
    elif scheduler_type == 'Heun':
        return HeunDiscreteScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
    elif scheduler_type == 'LMS':
        return LMSDiscreteScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
    elif scheduler_type == 'LMS Karras':
        return LMSDiscreteScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000, use_karras_sigmas=True)
    elif scheduler_type == 'DEIS':
        return DEISMultistepScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
    elif scheduler_type == 'UniPC':
        return UniPCMultistepScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
    else:
        raise ValueError("Unknown scheduler type")
# Define combinations of worst-performing classes
worst_combinations = [
    ['Impurity'],
    ['Regular crystal', 'Impurity'],
    ['Impurity', 'Bubbles'],
    ['Impurity', 'Droplets'],
    ['Regular crystal', 'Impurity', 'Bubbles', 'Droplets']
]

# Function to create combined prompts
def create_combined_prompt(combination):
    main_prompt = prompts[combination[0]]
    for label in combination[1:]:
        main_prompt += f" with {prompts[label].split('of ')[-1]}"
    return main_prompt

def crop_or_resize_image(image, target_size=(429, 600)):
    return ImageOps.fit(image, target_size, method=0, bleed=0.0, centering=(0.5, 0.5))

# Function to generate images using the LoRa model
def generate_image(pipeline, prompt, image_idx, output_dir, target_size=(429, 600)):
    try:
        result = pipeline(prompt)
        if result is None or not hasattr(result, 'images') or result.images is None:
            raise ValueError(f"No images returned for prompt: {prompt}")
        image = result.images[0]
        image = crop_or_resize_image(image, target_size)  # Ensure the image is the correct size
        image_path = os.path.join(output_dir, f"{prompt.replace(' ', '_').lower()}_{image_idx}.png")
        image.save(image_path)
        return image_path
    except Exception as e:
        print(f"Error generating image {image_idx} for prompt {prompt}: {e}")
        torch.cuda.empty_cache()  # Clear GPU cache to free up memory
        return None


# Initialize PartialState
distributed_state = PartialState()

# Load the DiffusionPipeline and move it to the appropriate device
device = distributed_state.device
scheduler = EulerAncestralDiscreteScheduler(beta_start=0.0001, beta_end=0.02, num_train_timesteps=1000)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16).to(device)
pipeline.scheduler = scheduler

lora_paths = {
    "Regular crystal": "/opt/data/AnSh/backup/lora/Regular_crystal_only/checkpoint-6500",
    "Impurity": "/opt/data/AnSh/backup/lora/Impurity_only",
    "Bubbles": "/opt/data/AnSh/backup/lora/Bubbles_only/checkpoint-1000",
    "Droplets": "/opt/data/AnSh/backup/lora/Droplets_only"
}

for label, path in lora_paths.items():
    pipeline.load_lora_weights(path, weight_name="pytorch_lora_weights.safetensors", adapter_name=label.lower().replace(' ', '_'))





prompt = "A microscopic image of a formation of regular_crystal"
# Generate image
pipeline.set_adapters(['regular_crystal'], adapter_weights=[0.6])
# Function to generate images using the LoRa model
def generate_image(pipeline, prompt, image_idx, output_dir, target_size):
    try:
        # Cast input to float16
        with torch.cuda.amp.autocast(dtype=torch.float16):
            result = pipeline(prompt,num_inference_steps=50, cross_attention_kwargs={"scale": 0.9},    height=1024,
    width=1024,guidance_scale=5)
        if result is None or not hasattr(result, 'images') or result.images is None:
            raise ValueError(f"No images returned for prompt: {prompt}")
        image = result.images[0]
        image = crop_or_resize_image(image, target_size)  # Ensure the image is the correct size
        image_path = os.path.join(output_dir, f"{prompt.replace(' ', '_').lower()}_{image_idx}.png")
        image.save(image_path)
        return image_path
    except Exception as e:
        print(f"Error generating image {image_idx} for prompt {prompt}: {e}")
        torch.cuda.empty_cache()  # Clear GPU cache to free up memory
        return None
image_paths = []
for i in range(1000):
    image_path = generate_image(output_dir="testing_images_regular_crystal",image_idx=i,pipeline=pipeline,prompt=prompt,  target_size=(429, 600))
    if image_path is not None:
        image_paths.append(image_path)
    # Create metadata
    metadata = []
    for idx, image_path in enumerate(image_paths):
        metadata.append({
            "Unnamed: 0": idx,
            "File_Path": image_path,
            "Image_ID": os.path.basename(image_path),
            "Solute": "Generated",
            "Solvent": "Generated",
            "Experiment_Number": 1,
            "Reactor_Label": "A",
            "Time_Stamp": f"{idx:06d}s",
            "Formatted_Time": f"0:00:{idx:02d}",
            "Helm_value": 0,
            "Temperatures": 0,
            "Transmissivity": 0,
            "Object present": 1,
            "Needle-like crystal": 0,
            "Elongated crystal": 0,
            "Platelet crystal": 0,
            "Regular crystal": 1,
            "Impurity": 0,
            "Agglomerated crystals": 0,
            "Bubbles": 0,
            "Droplets": 0,
            "Too concentrated": 0
        })
    
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv("testing_images_regular_crystal_metadata.csv", index=False)

    

