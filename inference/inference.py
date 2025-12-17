import torch
import numpy as np
from PIL import Image
import os
import pickle
from model import GPT, GPTConfig
from data_utils import load_palette, tokens_to_image
import time
import wandb

def generate_samples(model_path='checkpoints/teacher_final.pt', num_samples=5):
    # WandB Init for Inference
    wandb.init(project="speculative-decoding-distillation", name="teacher-inference", config={
        "num_samples": num_samples,
        "model_path": model_path
    })

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load Palette
    palette = load_palette()
    
    # Load Model
    config = GPTConfig(
        vocab_size=512,
        block_size=4096,
        n_layer=12,
        n_head=8,
        n_embd=512
    )
    model = GPT(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    os.makedirs('generated_samples', exist_ok=True)
    
    print("Generating samples...")
    start_time = time.time()
    
    initial_idx = torch.randint(0, 512, (num_samples, 1)).to(device)
    
    with torch.no_grad():
        # Generate 4095 more tokens
        generated = model.generate(initial_idx, max_new_tokens=4095, temperature=1.0, top_k=100)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_sample = total_time / num_samples
    print(f"Generated {num_samples} samples in {total_time:.2f}s ({avg_time_per_sample:.2f}s per image)")
    
    wandb.log({
        "total_inference_time": total_time,
        "avg_time_per_sample": avg_time_per_sample,
        "samples_per_sec": num_samples / total_time
    })
    
    for i in range(num_samples):
        tokens = generated[i]
        save_path = f'generated_samples/sample_{i}.png'
        img = tokens_to_image(tokens, palette)
        img.save(save_path)
        print(f"Saved {save_path}")
        
        # Log image to wandb
        wandb.log({f"generated_sample_{i}": wandb.Image(img)})

    wandb.finish()

if __name__ == "__main__":
    generate_samples()
