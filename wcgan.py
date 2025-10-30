# wcgan.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Generator(nn.Module):
    def __init__(self, latent_dim, out_dim, hidden_dims=(256, 512), use_layernorm=True):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.Tanh())  # assume data scaled to [-1,1]
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dims=(512, 256), dropout=0.3, use_spectral=False):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            lin = nn.Linear(prev, h)
            if use_spectral:
                lin = nn.utils.spectral_norm(lin)
            layers.append(lin)
            layers.append(nn.LeakyReLU(0.2))
            if dropout:
                layers.append(nn.Dropout(dropout))
            prev = h
        final = nn.Linear(prev, 1)
        if use_spectral:
            final = nn.utils.spectral_norm(final)
        layers.append(final)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).view(-1)


def gradient_penalty(D, real, fake, device, lambda_gp=10.0):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, *([1] * (real.dim() - 1)), device=device)
    alpha = alpha.expand_as(real)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = D(interpolates)
    grads = grad(outputs=d_interpolates, inputs=interpolates,
                 grad_outputs=torch.ones_like(d_interpolates, device=device),
                 create_graph=True, retain_graph=True)[0]
    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gp


class WCGAN_GP:
    def __init__(self, feature_dim, latent_dim=100, device=DEVICE):
        self.device = device
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.G = Generator(latent_dim, out_dim=feature_dim).to(device)
        self.D = Discriminator(in_dim=feature_dim).to(device)

    def sample_noise(self, n):
        return torch.randn(n, self.latent_dim, device=self.device)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.G.state_dict(), os.path.join(path, "generator.pth"))
        torch.save(self.D.state_dict(), os.path.join(path, "discriminator.pth"))

    def load(self, path):
        self.G.load_state_dict(torch.load(os.path.join(path, "generator.pth"), map_location=self.device))
        self.D.load_state_dict(torch.load(os.path.join(path, "discriminator.pth"), map_location=self.device))

    def generate(self, n_samples):
        self.G.eval()
        with torch.no_grad():
            z = self.sample_noise(n_samples)
            samples = self.G(z).cpu().numpy()
        return samples

    def train(self, X: np.ndarray, epochs=100, batch_size=64, lr=1e-4, beta1=0.0, beta2=0.9,
              n_critic=5, lambda_gp=10.0, log_interval=10, checkpoint_dir=None):
        """
        Train WGAN-GP on numpy array X (should be scaled to [-1,1]).
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        opt_G = optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))
        opt_D = optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))

        loss_history = {"g": [], "d": []}
        for epoch in range(epochs):
            for i, (real_batch,) in enumerate(loader):
                real = real_batch.to(self.device)

                # Train discriminator n_critic times
                for _ in range(n_critic):
                    z = self.sample_noise(real.size(0))
                    fake = self.G(z).detach()
                    opt_D.zero_grad()
                    d_real = self.D(real)
                    d_fake = self.D(fake)
                    gp = gradient_penalty(self.D, real, fake, self.device, lambda_gp=lambda_gp)
                    loss_D = -d_real.mean() + d_fake.mean() + gp
                    loss_D.backward()
                    opt_D.step()

                # Train generator
                z = self.sample_noise(real.size(0))
                opt_G.zero_grad()
                gen = self.G(z)
                loss_G = -self.D(gen).mean()
                loss_G.backward()
                opt_G.step()

            loss_history['g'].append(loss_G.item())
            loss_history['d'].append(loss_D.item())

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                print(f"[WCGAN] Epoch {epoch+1}/{epochs} | D_loss {loss_D.item():.4f} | G_loss {loss_G.item():.4f}")

            if checkpoint_dir and ((epoch + 1) % (log_interval * 5) == 0):
                self.save(checkpoint_dir)

        if checkpoint_dir:
            self.save(checkpoint_dir)
        return loss_history
