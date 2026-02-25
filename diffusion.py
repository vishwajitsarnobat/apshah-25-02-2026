import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib
# matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NoiseSchedule:
    def __init__(self, T=500, beta_start=1e-4, beta_end=0.02):
        self.T=T
        betas=torch.linspace(beta_start,beta_end,T)
        alphas=1-betas
        alpha_bar=torch.cumprod(alphas,dim=0)
        self.betas=betas.to(device)
        self.alphas=alphas.to(device)
        self.alpha_bar=alpha_bar.to(device)
        self.sqrt_ab=alpha_bar.sqrt().to(device)
        self.sqrt_one_m_ab=(1-alpha_bar).sqrt().to(device)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise=torch.randn_like(x0)
        sqrt_ab=self.sqrt_ab[t].view(-1,1,1,1) #pytorch wants in form (B,C,H,W)
        sqrt_1mab=self.sqrt_one_m_ab[t].view(-1,1,1,1)
        return sqrt_ab*x0+sqrt_1mab*noise, noise #new image, added noise
        
def visualize_forward_process(schedule, x0_sample, steps_to_show=None):
    if steps_to_show is None:
        steps_to_show = [0,50,100,200,300,400,schedule.T-1]
    fig, axes = plt.subplots(1, len(steps_to_show), figsize=(14,3))
    fig.suptitle("Forward Process: Clean->Pure Noise", fontsize=14, fontweight='bold')

    for ax, t_val in zip(axes, steps_to_show):
        t_tensor = torch.tensor([t_val],device=device)
        x_t, _ = schedule.q_sample(x0_sample, t_tensor)
        img = x_t[0,0].cpu().numpy()
        ax.imshow(img, cmap='grey', vmin=-1, vmax=1)
        ax.set_title(f't={t_val}', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("forward_process.png", dpi=120, bbox_inches='tight')
    # plt.show()
    print('Saved forward process...')

def plot_noise_schedule(schedule):
    t = np.arange(schedule.T)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Noise Schedule (Linear)", fontsize=13, fontweight='bold')
    axes[0].plot(t, schedule.sqrt_ab.cpu(), color='steelblue', label='sqrt(alpha_t)/signal')
    axes[0].plot(t, schedule.sqrt_one_m_ab.cpu(), color='tomato', label='sqrt(1-alpha(t))/noise')
    axes[0].set_xlabel("Timestep t")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title("Signal vs Noise Coefficients")
    axes[0].legend()
    axes[0].grid(alpha=0.3) #alpha 0 means full transparency
    snr = (schedule.sqrt_ab/schedule.sqrt_one_m_ab).cpu()
    axes[1].semilogy(t, snr, color='purple') #semilogy means logarithmic y axis
    axes[1].set_xlabel("Timestamp t")
    axes[1].set_ylabel("SNR (log scale)")
    axes[1].set_title("Signal-to-Noise ratio vs t")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("noise_schedule.png", dpi=120, bbox_inches='tight')
    # plt.show()
    print("Saved noise schedule...")

#from transformer paper
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim=dim

    def forward(self, t):
        half = self.dim//2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        emb = t[:, None].float() * freqs[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

#every block of unet
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch) #injecting t
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)

class TinyUNet(nn.Module):
    def __init__(self, time_dim=128, base_ch=32):
        super().__init__()
        C = base_ch
        self.time_emb = SinusoidalEmbedding(time_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim*2), nn.SiLU(), nn.Linear(time_dim*2, time_dim))
        
        #encoder
        self.enc1 = ResBlock(1, C, time_dim) # 28×28
        self.enc2 = ResBlock(C, C*2, time_dim) # 14×14
        self.enc3 = ResBlock(C*2, C*4, time_dim) # 7×7

        self.down1 = nn.MaxPool2d(2)
        self.down2 = nn.MaxPool2d(2)

        self.bot = ResBlock(C*4, C*4, time_dim)

        #decoder
        self.up2 = nn.ConvTranspose2d(C*4, C*2, 2, stride=2)
        self.dec2 = ResBlock(C*4, C*2, time_dim) #C*4 because of skip connections

        self.up1 = nn.ConvTranspose2d(C*2, C, 2, stride=2)
        self.dec1 = ResBlock(C*2, C, time_dim)

        self.out_conv = nn.Conv2d(C, 1, 1) #final 1x1 conv

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_emb(t))
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)
        b = self.bot(e3, t_emb)
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.out_conv(d1)

def train(model, schedule, loader, optimizer, n_epochs=20):
    model.train()
    losses = []
    print(f"Training DDPM for {n_epochs} and T = {schedule.T}")
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_idx, (x0, _) in enumerate(loader):
            x0 = x0.to(device)
            B = x0.size(0)
            t = torch.randint(0, schedule.T, (B,), device=device)
            x_t, noise = schedule.q_sample(x0, t) #noise addition
            eps_pred = model(x_t, t) #prediction
            loss = F.mse_loss(eps_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        if (epoch+1)%5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.5f}")
    
    print("Training completed!")
    return losses

def plot_loss(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, color='steelblue', linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Training Loss - Noise Prediction MSE")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=120)
    # plt.show()
    print("Saved loss_curve.png")

#reverse process
@torch.no_grad()
def ddpm_sample(model, schedule, n_samples=16, img_size=28, save_frames=True):
    model.eval()
    T = schedule.T
    x = torch.randn(n_samples, 1, img_size, img_size, device=device)
    frames = []
    steps_to_save = set(range(0, T, T//20)) | {T-1}
    print("Sampling (reverse denoising)")
    for t_val in reversed(range(T)):
        t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        eps_pred = model(x, t)
        alpha_t = schedule.alphas[t_val]
        ab_t = schedule.alpha_bar[t_val]
        ab_prev = schedule.alpha_bar[t_val - 1] if t_val > 0 else torch.tensor(1.0, device=device)
        beta_t = schedule.betas[t_val]
        x0_pred = (x-(1-ab_t).sqrt() * eps_pred) / ab_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)
        mean = (ab_prev.sqrt()*beta_t/(1-ab_t)) * x0_pred + (alpha_t.sqrt()*(1-ab_prev)/(1-ab_t)) * x
        if t_val > 0:
            noise = torch.randn_like(x)
            sigma = beta_t.sqrt()
            x = mean + sigma * noise
        else:
            x = mean
        if t_val in steps_to_save:
            frames.append(x.cpu().clone())
    return x.cpu(), frames

def show_samples(samples, title="Generated MNIST Samples"):
    n = int(samples.shape[0] ** 0.5)
    fig, axes = plt.subplots(n, n, figsize=(8, 8))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("generated_samples.png", dpi=120, bbox_inches='tight')
    # plt.show()
    print("Saved generated_samples.png")

def show_denoising_progression(frames, sample_idx=0):
    n = len(frames)
    grid = int(np.ceil(np.sqrt(n)))  # square grid
    fig, axes = plt.subplots(grid, grid, figsize=(8, 8))
    fig.suptitle(f"Denoising Progression (Sample #{sample_idx}): Noise → Digit", fontsize=12)
    axes = axes.flatten()
    for i in range(grid * grid):
        ax = axes[i]
        if i < n:
            frame = frames[i]
            ax.imshow(frame[sample_idx, 0].numpy(), cmap='gray', vmin=-1, vmax=1)
            ax.set_title(f"step {i}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("denoising_progression.png", dpi=120, bbox_inches='tight')
    print("Saved denoising_progression.png")

def main():
    print("Training DDPM on MNIST")
    # Hyperparameters
    TS        = 500
    N_EPOCHS  = 20
    BATCH     = 128
    LR        = 3e-4
    print("Loading MNIST...")
    transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: x * 2 - 1)])  # [0,1]→[-1,1]
    dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    schedule = NoiseSchedule(T=TS)
    print("Visualizing forward process...")
    x0_sample, _ = next(iter(loader))
    x0_sample = x0_sample[:1].to(device)
    visualize_forward_process(schedule, x0_sample)
    plot_noise_schedule(schedule)
    print("Building Tiny U-Net...")
    model = TinyUNet(time_dim=128, base_ch=32).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print("Training DDPM...")
    losses = train(model, schedule, loader, optimizer, n_epochs=N_EPOCHS)
    plot_loss(losses)
    print("Sampling from noise...")
    samples, frames = ddpm_sample(model, schedule, n_samples=16)
    print("Done!")
    show_samples(samples)
    show_denoising_progression(frames, sample_idx=0)

if __name__ == "__main__":
    main()