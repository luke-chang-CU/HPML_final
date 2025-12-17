from data_utils import MiniImageNetDataset
import torch

def test_dataset():
    print("Initializing dataset...")
    dataset = MiniImageNetDataset(root_dir='dataset', split='train', build_palette=False, return_image=True)
    print(f"Dataset size: {len(dataset)}")
    
    print("Getting item 0...")
    item = dataset[0]
    print(f"Item shape: {item.shape}, Type: {item.dtype}, Min: {item.min()}, Max: {item.max()}")
    
    print("Dataset test passed.")

if __name__ == "__main__":
    test_dataset()
