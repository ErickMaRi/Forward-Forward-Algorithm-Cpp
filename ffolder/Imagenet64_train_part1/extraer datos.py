import os
import pickle
import numpy as np
from PIL import Image

# Define paths
batch_files = ['train_data_batch_2']
output_folder = 'output_images'

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each batch file and extract images
for batch_file in batch_files:
    with open(batch_file, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        images = batch['data']
        labels = batch['labels']
        
        # Reshape and save images
        for i in range(10000):
            img = images[i].reshape(3, 64, 64).transpose(1, 2, 0)  # Reshape to HWC format
            img = Image.fromarray(np.uint8(img))  # Convert to PIL Image
            img.save(os.path.join(output_folder, f'image_{i + len(images) * batch_files.index(batch_file)}.png'))

        print(f'Processed {len(images)} images from {batch_file}')