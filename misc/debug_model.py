import torch
from vqvae import VQVAE

def test_model():
    print("Initializing model...")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    model = VQVAE(num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32,
                  num_embeddings=1024, embedding_dim=64, commitment_cost=0.25)
    model.to('cuda')
    print("Model initialized.")
    
    dummy_input = torch.randn(2, 3, 64, 64).to('cuda')
    print("Forward pass...")
    loss, x_recon, perplexity = model(dummy_input)
    print(f"Output shape: {x_recon.shape}")
    print("Model test passed.")

if __name__ == "__main__":
    test_model()
