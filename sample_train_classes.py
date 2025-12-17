
import os
import torch
import numpy as np
from PIL import Image
from data_utils import MiniImageNetDataset
import tqdm

def sample_classes():
    # Initialize Dataset
    print("Loading Training Dataset...")
    dataset = MiniImageNetDataset(split='train', return_image=True)
    
    output_dir = 'train_class_samples'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Sampling 1 image from each of the {len(dataset.classes)} classes...")
    
    # Iterate through all classes
    # We know there are 64 classes in train, indices 0 to 63
    # data_utils.py builds class_to_idx sorted.
    
    for class_idx in range(len(dataset.classes)):
        class_name = dataset.classes[class_idx]
        
        # Get random image tensor (3, 64, 64)
        img_tensor = dataset.get_class_image(class_idx)
        
        # Convert to unit8 numpy
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # Save
        filename = f"class_{class_idx:02d}_{class_name}.png"
        save_path = os.path.join(output_dir, filename)
        img_pil.save(save_path)
        
    print(f"Saved {len(dataset.classes)} images to {output_dir}/")

if __name__ == "__main__":
    sample_classes()
