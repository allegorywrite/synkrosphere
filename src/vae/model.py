"""
VAE (Variational Autoencoder) モデル定義
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder モデル
    画像生成のための基本的なVAEモデル
    """
    
    def __init__(self, input_dim=3, img_size=256, latent_dim=64):
        """
        Args:
            input_dim: 入力チャンネル数 (RGB=3)
            img_size: 入力画像サイズ (正方形を想定)
            latent_dim: 潜在空間の次元数
        """
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(256 * (img_size // 16) * (img_size // 16), latent_dim)
        self.fc_logvar = nn.Linear(256 * (img_size // 16) * (img_size // 16), latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, 256 * (img_size // 16) * (img_size // 16))
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_dim, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 出力を[0,1]に正規化
        )
    
    def encode(self, x):
        """入力画像を潜在ベクトルにエンコード"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """再パラメータ化トリック"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """潜在ベクトルから画像を生成"""
        h = self.decoder_input(z)
        h = h.view(-1, 256, self.img_size // 16, self.img_size // 16)
        return self.decoder(h)
    
    def forward(self, x):
        """順伝播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def sample(self, num_samples, device='cuda'):
        """潜在空間からランダムサンプリング"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    
    def interpolate(self, z1, z2, steps=10):
        """2つの潜在ベクトル間を補間"""
        z_interp = torch.zeros(steps, self.latent_dim, device=z1.device)
        for i in range(steps):
            t = i / (steps - 1)
            z_interp[i] = z1 * (1 - t) + z2 * t
        return self.decode(z_interp)
