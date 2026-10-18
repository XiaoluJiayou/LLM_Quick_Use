import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
latent_dim = 16  # 潜在空间维度
image_size = 28  # MNIST 图像大小
channels = 1     # MNIST 单通道
timesteps = 1000 # 扩散步数
batch_size = 64
epochs = 10

def create_train_loader(data_dir="./data", batch_size=batch_size, download=False):
    """Dataset I/O is explicit; importing this module never downloads data."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=download)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 自动编码器定义
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, 4, stride=2, padding=1),  # [batch, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, latent_dim, 4, stride=2, padding=1),  # [batch, latent_dim, 7, 7]
            nn.ReLU()
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 16, 4, stride=2, padding=1),  # [batch, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, 4, stride=2, padding=1),  # [batch, 1, 28, 28]
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# UNet 简化版（用于去噪）
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(latent_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [batch, 64, 4, 4]
            nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),  # [batch, 32, 7, 7]
            nn.ReLU(),
            nn.Conv2d(32, latent_dim, 3, padding=1)
        )
        # 时间嵌入
        self.time_embed = nn.Embedding(timesteps, 64)

    def forward(self, x, t):
        t_emb = self.time_embed(t).view(-1, 64, 1, 1)  # [batch, 64, 1, 1]
        x = self.down(x)
        x = x + t_emb  # 简单的时间条件注入
        x = self.up(x)
        return x

# 扩散过程工具函数
class Diffusion:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)  # 线性噪声调度
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        """前向过程：添加噪声"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise, noise

    def p_sample(self, model, x, t):
        """后向过程：去噪一步"""
        t = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        noise_pred = model(x, t)
        alpha = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        one_minus_alpha_cumprod = 1.0 - alpha_cumprod
        sqrt_one_minus_alpha_cumprod = torch.sqrt(one_minus_alpha_cumprod)
        posterior_mean = (x - (1 - alpha) / sqrt_one_minus_alpha_cumprod * noise_pred) / torch.sqrt(alpha)
        if t[0] > 0:
            noise = torch.randn_like(x)
            return posterior_mean + torch.sqrt(1 - alpha) * noise
        return posterior_mean

# 训练自动编码器
def train_autoencoder(ae, optimizer, train_loader, epochs=5):
    ae.train()
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon, _ = ae(data)
            loss = F.mse_loss(recon, data)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"AE Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# 训练 LDM
def train_ldm(ae, unet, diffusion, optimizer, train_loader, epochs=epochs):
    unet.train()
    ae.eval()  # 固定自动编码器
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            with torch.no_grad():
                _, z = ae(data)  # 获取潜在表示
            t = torch.randint(0, diffusion.timesteps, (data.size(0),), device=device)
            z_noisy, noise = diffusion.q_sample(z, t)
            optimizer.zero_grad()
            noise_pred = unet(z_noisy, t)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"LDM Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# 生成样本
def sample_ldm(ae, unet, diffusion, n_samples=16):
    unet.eval()
    ae.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, latent_dim, 7, 7).to(device)  # 从噪声开始
        for t in reversed(range(diffusion.timesteps)):
            x = diffusion.p_sample(unet, x, t)
        images = ae.decoder(x)
    return images

def main(argv=None):
    import argparse
    global device, timesteps
    parser = argparse.ArgumentParser(description="Educational MNIST latent diffusion demo")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ae-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--device", default=str(device))
    parser.add_argument("--smoke-test", action="store_true", help="Synthetic forward/backward, no dataset or download")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.ae_epochs, args.batch_size, args.timesteps) < 1:
        parser.error("epoch counts, batch-size and timesteps must be positive")
    device, timesteps = torch.device(args.device), args.timesteps
    torch.manual_seed(0)
    ae, unet = Autoencoder().to(device), SimpleUNet().to(device)
    diffusion = Diffusion(timesteps)
    if args.smoke_test:
        data = torch.randn(2, channels, image_size, image_size, device=device)
        recon, z = ae(data)
        t = torch.zeros(2, dtype=torch.long, device=device)
        noisy, noise = diffusion.q_sample(z.detach(), t)
        prediction = unet(noisy, t)
        assert prediction.shape == noise.shape
        (F.mse_loss(prediction, noise) + F.mse_loss(recon, data)).backward()
        print("OK: reconstruction={}, denoiser={}".format(tuple(recon.shape), tuple(prediction.shape)))
        return 0
    loader = create_train_loader(args.data_dir, args.batch_size, args.download)
    train_autoencoder(ae, optim.Adam(ae.parameters(), lr=1e-3), loader, epochs=args.ae_epochs)
    train_ldm(ae, unet, diffusion, optim.Adam(unet.parameters(), lr=1e-4), loader, epochs=args.epochs)
    samples = sample_ldm(ae, unet, diffusion).cpu().numpy()
    print("Sample shape:", samples.shape)
    if not args.no_show:
        import matplotlib.pyplot as plt
        _, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i, 0], cmap="gray")
            ax.axis("off")
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
