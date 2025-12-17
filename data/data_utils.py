import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import pickle

from datasets import load_dataset

class MiniImageNetDataset(Dataset):
    def __init__(self, root_dir='dataset', split='train', transform=None, return_image=True, build_palette=False, return_label=False):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.return_image = return_image
        self.return_label = return_label
        
        # Load CSV
        csv_path = os.path.join(root_dir, f'{split}.csv')
        self.data_df = []
        with open(csv_path, 'r') as f:
            lines = f.readlines()[1:] # Skip header
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    self.data_df.append((parts[0], parts[1]))

        # Build class mapping from TRAIN split always to be consistent
        train_csv_path = os.path.join(root_dir, 'train.csv')
        unique_labels = set()
        with open(train_csv_path, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    unique_labels.add(parts[1])
        
        self.classes = sorted(list(unique_labels))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_dir = os.path.join(root_dir, 'images')
        print(f"Loaded {len(self.data_df)} images from {csv_path}. Found {len(self.classes)} unique classes.")
        
        # Build index for fast retrieval of images by class
        self.label_idx_to_indices = {}
        # Only build this if we have a valid class_to_idx (which comes from train)
        # Scan data_df once
        for idx, (_, label_str) in enumerate(self.data_df):
            label_idx = self.class_to_idx.get(label_str, -1)
            if label_idx != -1:
                if label_idx not in self.label_idx_to_indices:
                    self.label_idx_to_indices[label_idx] = []
                self.label_idx_to_indices[label_idx].append(idx)

    def get_class_image(self, class_idx):
        if class_idx not in self.label_idx_to_indices:
            print(f"Class index {class_idx} not found in this split.")
            # Return black image
            return torch.zeros(3, 64, 64)
            
        indices = self.label_idx_to_indices[class_idx]
        random_idx = np.random.choice(indices)
        
        # Use __getitem__ but force return_image behavior
        # We need to temporarily ensure we get an image, not just token/label
        # But __getitem__ depends on self.return_image flags.
        # Let's just manually load it to avoid messing with flags or recursion.
        
        filename, _ = self.data_df[random_idx]
        img_path = os.path.join(self.image_dir, filename)
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize((64, 64))
            img_tensor = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
            return img_tensor
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 64, 64)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        filename, label_str = self.data_df[idx]
        img_path = os.path.join(self.image_dir, filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (84, 84))

        # Original is 84x84, resize to 64x64 for VQ-VAE
        image = image.resize((64, 64))
        
        # (C, H, W) in [0, 1]
        img_tensor = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        label_idx = self.class_to_idx.get(label_str, -1) # -1 for unknown classes (e.g. val/test not in train)

        if self.return_image:
            if self.return_label:
                return img_tensor, label_idx
            return img_tensor
            
        # If returning tokens/palette (legacy logic removed/ignored), just return tensor
        if self.return_label:
            return img_tensor, label_idx
        return img_tensor


class TokenDataset(Dataset):
    def __init__(self, token_file):
        data = torch.load(token_file)
        if isinstance(data, dict):
            self.tokens = data['tokens']
            self.labels = data['labels']
            print(f"Loaded {len(self.tokens)} sequences from {token_file}")
            # Only claim we have labels if they are not None
            if self.labels is not None:
                self.has_labels = True
                print("Labels found.")
            else:
                self.has_labels = False
                print("No labels found (Unconditional).")
        else:
            self.tokens = data
            self.labels = None
            print(f"Loaded {len(self.tokens)} sequences from {token_file} (No labels)")
            self.has_labels = False

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.has_labels:
            return self.tokens[idx], self.labels[idx]
        return self.tokens[idx]

if __name__ == "__main__":
    # Test and build palette
    dataset = MiniImageNetDataset(split='train', build_palette=True)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample values: {sample[:10]}")

def load_palette(palette_path='palette.pkl'):
    with open(palette_path, 'rb') as f:
        kmeans = pickle.load(f)
    return kmeans.cluster_centers_

def tokens_to_image(tokens, palette):
    # tokens: (1024,)
    # palette: (512, 3)
    
    # Map tokens to RGB values
    pixels = palette[tokens.cpu().numpy()]
    pixels = pixels.reshape(64, 64, 3).astype(np.uint8)
    
    img = Image.fromarray(pixels)
    return img
