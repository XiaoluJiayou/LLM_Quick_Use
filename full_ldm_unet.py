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
channels = 1     # 单通道
timesteps = 1000 # 扩散步数
batch_size = 64
epochs = 10
num_classes = 10  # MNIST 类别数

def create_train_loader(data_dir="./data", batch_size=batch_size, download=False):
    """Dataset I/O is explicit; importing this module never downloads data."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=download)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 跨注意力模块
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=4, dim_head=64):
        super(CrossAttention, self).__init__()
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        # x: [batch, channels, height, width] -> [batch, height*width, channels]
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)  # [batch, hw, channels]

        # 多头注意力
        q = self.to_q(x)  # [batch, hw, inner_dim]
        k = self.to_k(context)  # [batch, context_len, inner_dim]
        v = self.to_v(context)  # [batch, context_len, inner_dim]

        q = q.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)  # [batch, heads, hw, dim_head]
        k = k.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)  # [batch, heads, context_len, dim_head]
        v = v.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)  # [batch, heads, context_len, dim_head]

        # 注意力计算
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [batch, heads, hw, context_len]
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [batch, heads, hw, dim_head]

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.dim_head)  # [batch, hw, inner_dim]
        out = self.to_out(out)  # [batch, hw, channels]
        out = out.permute(0, 2, 1).view(batch_size, channels, height, width)  # [batch, channels, h, w]
        return out

# 条件 UNet
class ConditionalUNet(nn.Module):
    def __init__(self):
        super(ConditionalUNet, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(latent_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [batch, 64, 4, 4]
            nn.ReLU()
        )
        self.time_embed = nn.Embedding(timesteps, 64)
        self.context_embed = nn.Embedding(num_classes, 64)  # 条件编码器 tau_theta
        self.attn = CrossAttention(query_dim=64, context_dim=64)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),  # [batch, 32, 7, 7]
            nn.ReLU(),
            nn.Conv2d(32, latent_dim, 3, padding=1)
        )

    def forward(self, x, t, y):
        t_emb = self.time_embed(t).view(-1, 64, 1, 1)  # [batch, 64, 1, 1]
        y_emb = self.context_embed(y)  # [batch, 64], tau_theta(y)

        x = self.down(x)  # [batch, 64, 4, 4]
        x = x + t_emb  # 注入时间条件
        x = x + self.attn(x, y_emb.unsqueeze(1))  # 跨注意力融合条件信息
        x = self.up(x)
        return x

# 自动编码器（简化为无条件版本）
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, latent_dim, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# 扩散过程
class Diffusion:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise, noise

    def p_sample(self, model, x, t, y):
        t = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        noise_pred = model(x, t, y)
        alpha = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
        posterior_mean = (x - (1 - alpha) / sqrt_one_minus_alpha_cumprod * noise_pred) / torch.sqrt(alpha)
        if t[0] > 0:
            noise = torch.randn_like(x)
            return posterior_mean + torch.sqrt(1 - alpha) * noise
        return posterior_mean

# 训练函数
def train_ldm(ae, unet, diffusion, optimizer, train_loader, epochs=epochs):
    unet.train()
    ae.eval()
    for epoch in range(epochs):
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            with torch.no_grad():
                _, z = ae(data)
            t = torch.randint(0, diffusion.timesteps, (data.size(0),), device=device)
            z_noisy, noise = diffusion.q_sample(z, t)
            optimizer.zero_grad()
            noise_pred = unet(z_noisy, t, labels)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# 生成样本
def sample_ldm(ae, unet, diffusion, y, n_samples=16):
    unet.eval()
    ae.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, latent_dim, 7, 7).to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device).reshape(-1)
        if y.numel() == 1:
            y = y.repeat(n_samples)
        if y.numel() != n_samples or bool(((y < 0) | (y >= num_classes)).any()):
            raise ValueError("Provide one class label or one valid class label per sample")
        for t in reversed(range(diffusion.timesteps)):
            x = diffusion.p_sample(unet, x, t, y)
        images = ae.decoder(x)
    return images

def main(argv=None):
    import argparse
    global device, timesteps
    parser = argparse.ArgumentParser(description="Educational conditional MNIST latent diffusion demo")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ae-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--class-label", type=int, default=5)
    parser.add_argument("--device", default=str(device))
    parser.add_argument("--smoke-test", action="store_true", help="Synthetic forward/backward, no dataset or download")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.ae_epochs, args.batch_size, args.timesteps) < 1 or not 0 <= args.class_label < num_classes:
        parser.error("Counts must be positive and class-label must be 0..9")
    device, timesteps = torch.device(args.device), args.timesteps
    torch.manual_seed(0)
    ae, unet = Autoencoder().to(device), ConditionalUNet().to(device)
    diffusion = Diffusion(timesteps)
    if args.smoke_test:
        data = torch.randn(2, channels, image_size, image_size, device=device)
        recon, z = ae(data)
        t = torch.zeros(2, dtype=torch.long, device=device)
        labels = torch.tensor([1, 2], dtype=torch.long, device=device)
        noisy, noise = diffusion.q_sample(z.detach(), t)
        prediction = unet(noisy, t, labels)
        assert prediction.shape == noise.shape
        (F.mse_loss(prediction, noise) + F.mse_loss(recon, data)).backward()
        print("OK: reconstruction={}, conditional_denoiser={}".format(tuple(recon.shape), tuple(prediction.shape)))
        return 0
    loader = create_train_loader(args.data_dir, args.batch_size, args.download)
    ae_optimizer = optim.Adam(ae.parameters(), lr=1e-3)
    ae.train()
    for _ in range(args.ae_epochs):
        for data, _ in loader:
            data = data.to(device)
            ae_optimizer.zero_grad()
            recon, _ = ae(data)
            F.mse_loss(recon, data).backward()
            ae_optimizer.step()
    train_ldm(ae, unet, diffusion, optim.Adam(unet.parameters(), lr=1e-4), loader, epochs=args.epochs)
    samples = sample_ldm(ae, unet, diffusion, args.class_label).cpu().numpy()
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
