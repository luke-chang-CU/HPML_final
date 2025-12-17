import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Standard Vector Quantization Layer.

    Args:
        num_embeddings (int): Vocabulary size (number of codebook vectors).
        embedding_dim (int): Dimensionality of each codebook vector.
        commitment_cost (float): Beta term for the commitment loss (scales encoder penalty).
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # inputs: (B, C, H, W) -> (B, H, W, C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # (B, H, W, C) -> (B, C, H, W)
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([
            Residual(in_channels, num_hiddens, num_residual_hiddens)
            for _ in range(self._num_residual_layers)
        ])
    
    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return x

class Encoder(nn.Module):
    """
    VQ-VAE Encoder.
    
    Downsamples the 64x64 input image to a 16x16 grid of latent vectors.
    Architecture:
    - 2 Strided Convs (downsample 2x each)
    - 1 Conv
    - Residual Stack
    """
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        # 64x64 -> 32x32 -> 16x16
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)

class Decoder(nn.Module):
    """
    VQ-VAE Decoder.
    
    Upsamples the 16x16 quantized grid back to 64x64 image.
    Architecture:
    - 1 Conv
    - Residual Stack
    - 2 Transposed Convs (upsample 2x each)
    """
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        # 16x16 -> 32x32 -> 64x64
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return torch.sigmoid(self._conv_trans_2(x))

class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder (VQ-VAE).
    
    Compresses high-dimensional images into discrete tokens.
    
    Attributes:
        _encoder (Encoder): Compress image to latents.
        _vq_vae (VectorQuantizer): Quantize latents to nearest codebook vector.
        _decoder (Decoder): Reconstruct image from quantized latents.
    """
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                       commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity
    
    def encode(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        _, _, _, indices = self._vq_vae(z)
        
        # Flatten to (B, H*W)
        return indices.view(x.shape[0], -1)

    def decode(self, indices):
        # indices: (B, H*W)
        # map to embeddings
        # (B, H*W) -> (B, H, W)
        H = int(indices.shape[1]**0.5)
        W = H
        
        # Get embeddings
        # self._vq_vae._embedding(indices) -> (B, H*W, D)
        quantized = self._vq_vae._embedding(indices)
        quantized = quantized.view(indices.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        return self._decoder(quantized)
