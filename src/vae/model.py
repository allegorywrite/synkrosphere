"""
VAE (Variational Autoencoder) モデル定義
高品質なVAEモデルをAntixK/PyTorch-VAEから統合
"""

import torch
from .models.vanilla_vae import VanillaVAE

class VAE:
    """
    Variational Autoencoder モデルのラッパークラス
    AntixK/PyTorch-VAEの高品質な実装を使用
    """
    
    def __init__(self, input_dim=3, img_size=256, latent_dim=64):
        """
        Args:
            input_dim: 入力チャンネル数 (RGB=3)
            img_size: 入力画像サイズ (正方形を想定)
            latent_dim: 潜在空間の次元数
        """
        self.model = VanillaVAE(
            in_channels=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[32, 64, 128, 256, 512]
        )
        
        self.input_dim = input_dim
        self.img_size = img_size
        self.latent_dim = latent_dim
    
    def encode(self, x):
        """入力画像を潜在ベクトルにエンコード"""
        mu, log_var = self.model.encode(x)
        return mu, log_var
    
    def decode(self, z):
        """潜在ベクトルから画像を生成"""
        return self.model.decode(z)
    
    def forward(self, x):
        """順伝播"""
        result = self.model.forward(x)
        return result[0], result[2], result[3]
    
    def sample(self, num_samples, device='cuda'):
        """潜在空間からランダムサンプリング"""
        return self.model.sample(num_samples, current_device=device)
    
    def interpolate(self, z1, z2, steps=10):
        """2つの潜在ベクトル間を補間"""
        z_interp = torch.zeros(steps, self.latent_dim, device=z1.device)
        for i in range(steps):
            t = i / (steps - 1)
            z_interp[i] = z1 * (1 - t) + z2 * t
        return self.decode(z_interp)
