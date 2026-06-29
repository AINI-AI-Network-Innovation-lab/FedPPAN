import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalVariationalBottleneck(nn.Module):
    """Convolutional variational bottleneck with Gaussian reparameterization."""

    def __init__(self, in_channels: int, scale: float = 0.5, kernel_size: int = 5):
        super().__init__()
        latent_channels = max(1, int(in_channels * scale))
        padding = kernel_size // 2
        self.encoder = nn.Conv2d(
            in_channels, latent_channels * 2, kernel_size=kernel_size, padding=padding
        )
        self.decoder = nn.Conv2d(
            latent_channels, in_channels, kernel_size=kernel_size, padding=padding
        )
        self.last_kl = torch.tensor(0.0)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = self.encoder(x)
        mu, logvar = torch.chunk(stats, 2, dim=1)
        z = self._reparameterize(mu, logvar)
        out = self.decoder(z)
        self.last_kl = 0.5 * torch.mean(
            torch.square(mu) + torch.exp(logvar) - 1.0 - logvar
        )
        return out


class CVBNet(nn.Module):
    """LeNet-style classifier with early CVB placement (P=1)."""

    def __init__(self, cvb_scale: float = 0.5, cvb_kernel_size: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.cvb = ConvolutionalVariationalBottleneck(
            in_channels=32, scale=cvb_scale, kernel_size=cvb_kernel_size
        )
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.cvb(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_kl_loss(self) -> torch.Tensor:
        return self.cvb.last_kl


class CVBCNN4(nn.Module):
    """CNN4 variant with early CVB insertion for CIFAR-like inputs."""

    def __init__(self, num_classes: int = 10, cvb_scale: float = 0.5, cvb_kernel_size: int = 5):
        super(CVBCNN4, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.cvb = ConvolutionalVariationalBottleneck(
            in_channels=32, scale=cvb_scale, kernel_size=cvb_kernel_size
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cvb(x)
        x = self.pool(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_kl_loss(self) -> torch.Tensor:
        return self.cvb.last_kl
