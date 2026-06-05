import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
image_size = 28  # MNIST 图像大小
channels = 1     # 单通道
latent_dim = 16  # 潜在空间通道数
batch_size = 64
epochs = 10
lambda_adv = 1.0  # 对抗损失权重
lambda_kl = 1e-6  # KL 正则化权重

# 数据加载 (MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 自动编码器定义
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器：输出均值和方差
        self.enc_conv = nn.Sequential(
            nn.Conv2d(channels, 16, 4, stride=2, padding=1),  # [batch, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),  # [batch, 32, 7, 7]
            nn.ReLU()
        )
        self.enc_mu = nn.Conv2d(32, latent_dim, 3, padding=1)  # [batch, latent_dim, 7, 7]
        self.enc_logvar = nn.Conv2d(32, latent_dim, 3, padding=1)  # [batch, latent_dim, 7, 7]

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 16, 4, stride=2, padding=1),  # [batch, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, 4, stride=2, padding=1),  # [batch, 1, 28, 28]
            nn.Tanh()
        )

    def encode(self, x):
        h = self.enc_conv(x)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

# 判别器定义
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 4, stride=2, padding=1),  # [batch, 32, 14, 14]
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [batch, 64, 7, 7]
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 3, padding=1),  # [batch, 1, 7, 7]
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

# 感知损失（使用预训练 VGG）
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        self.layers = nn.Sequential(*list(vgg.children())[:9])  # 提取前几层
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # MNIST 是单通道，VGG 需要 3 通道，简单重复
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        x_feat = self.layers(x)
        y_feat = self.layers(y)
        return F.mse_loss(x_feat, y_feat)

# 训练函数
def train_autoencoder(ae, disc, percept_loss, ae_optimizer, disc_optimizer):
    ae.train()
    disc.train()
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            batch_size = data.size(0)

            # 训练判别器
            disc_optimizer.zero_grad()
            recon, mu, logvar, z = ae(data)
            real_pred = disc(data)
            fake_pred = disc(recon.detach())
            disc_loss = -torch.mean(torch.log(real_pred + 1e-8) + torch.log(1 - fake_pred + 1e-8))
            disc_loss.backward()
            disc_optimizer.step()

            # 训练自动编码器
            ae_optimizer.zero_grad()
            recon, mu, logvar, z = ae(data)
            fake_pred = disc(recon)

            # 计算损失
            perc_loss = percept_loss(data, recon)
            adv_loss = -torch.mean(torch.log(fake_pred + 1e-8))  # 生成器目标
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = perc_loss + lambda_adv * adv_loss + lambda_kl * kl_loss

            total_loss.backward()
            ae_optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}], "
                      f"Total Loss: {total_loss.item():.4f}, Perc: {perc_loss.item():.4f}, "
                      f"Adv: {adv_loss.item():.4f}, KL: {kl_loss.item():.4f}")

# 主程序
if __name__ == "__main__":
    # 初始化模型
    ae = Autoencoder().to(device)
    disc = Discriminator().to(device)
    percept_loss = PerceptualLoss().to(device)

    # 优化器
    ae_optimizer = optim.Adam(ae.parameters(), lr=1e-3)
    disc_optimizer = optim.Adam(disc.parameters(), lr=1e-3)

    # 训练
    print("Training Autoencoder with Perceptual, Adversarial, and KL Loss...")
    train_autoencoder(ae, disc, percept_loss, ae_optimizer, disc_optimizer)

    # 测试重建（可选）
    ae.eval()
    with torch.no_grad():
        data, _ = next(iter(train_loader))
        data = data.to(device)
        recon, _, _, z = ae(data)
        print("Latent z shape:", z.shape)  # [batch, latent_dim, 7, 7]
        print("Reconstructed shape:", recon.shape)  # [batch, 1, 28, 28]
