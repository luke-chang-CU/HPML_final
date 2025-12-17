import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from vqvae import VQVAE
from data_utils import MiniImageNetDataset
import os
from PIL import Image
import numpy as np

def verify_teacher():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load Teacher
    print("Loading Teacher...")
    config = GPTConfig(
        vocab_size=1024,
        block_size=256,
        n_layer=20,
        n_head=16,
        n_embd=1024,
        num_classes=64
    )
    model = GPT(config)
    
    ckpt_path = 'checkpoints_teacher_vqvae/teacher_final.pt'
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found!")
        return
        
    state_dict = torch.load(ckpt_path, map_location=device)
    # Fix for compiled model saving
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    model = torch.compile(model)
    
    # 2. Load VQ-VAE
    print("Loading VQ-VAE...")
    vqvae = VQVAE(num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=256,
                  num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
    
    vqvae_path = 'checkpoints_vqvae_miniimagenet/vqvae_final.pt'
    if os.path.exists(vqvae_path):
        vqvae.load_state_dict(torch.load(vqvae_path, map_location='cpu'))
    
    vqvae.to('cpu') # Decode on CPU to save VRAM or just simplicity
    vqvae.eval()
    
    # 3. Load Dataset for Real Images
    print("Loading Dataset for reference images...")
    dataset = MiniImageNetDataset(split='train', return_image=True)
    
    # 4. Generate Samples
    # Classes to test: Class 46 (requested), and 3 random others
    classes_to_test = [46] + torch.randint(0, 64, (3,)).tolist()
    
    print(f"Generating samples for classes: {classes_to_test}")
    
    # Prepare inputs
    # Shape: (4, 1)
    initial_idx = torch.randint(0, 1024, (4, 1)).to(device)
    class_labels = torch.tensor(classes_to_test, device=device)
    
    with torch.no_grad():
        generated = model.generate(initial_idx, max_new_tokens=255, temperature=1.0, top_k=100, class_labels=class_labels)
        
    # Decode
    generated_cpu = generated.cpu()
    
    output_dir = 'verification_samples'
    os.makedirs(output_dir, exist_ok=True)
    
    for i, class_idx in enumerate(classes_to_test):
        # Decode generated info
        indices = generated_cpu[i].view(1, 256)
        decoded = vqvae.decode(indices)
        img_gen = decoded[0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
        
        # Get real image
        img_real = dataset.get_class_image(class_idx) # (3, 64, 64) -> (64, 64, 3) needed
        img_real = img_real.permute(1, 2, 0).clamp(0, 1)
        
        # Concatenate side-by-side
        # (64, 128, 3)
        combined = torch.cat((img_gen, img_real), dim=1)
        
        # Save
        img_np = (combined.numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        save_path = os.path.join(output_dir, f'class_{class_idx}_verification.png')
        img_pil.save(save_path)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    verify_teacher()
