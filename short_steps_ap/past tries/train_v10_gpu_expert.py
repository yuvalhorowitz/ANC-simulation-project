import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

# --- 1. GPU & Precision Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS, BATCH_SIZE, EPOCHS = 8000, 32, 100
WINDOW_SIZE = 512

# Use updated Amp syntax for 2026
scaler = torch.amp.GradScaler('cuda')

# --- 2. Model Architecture: Resolution-Isolated Experts ---
class SpectralExpertTCN(nn.Module):
    def __init__(self):
        super(SpectralExpertTCN, self).__init__()
        
        # Expert 1: 50-400Hz (Drone Power) - Kernel 15 + Tanh
        self.drone_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1), nn.Tanh()
        )
        
        # Expert 2: 400-1200Hz (Harmonics) - Kernel 7 + Tanh
        self.mid_branch = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=7, padding=3),
            nn.BatchNorm1d(12), nn.ReLU(),
            nn.Conv1d(12, 1, kernel_size=1), nn.Tanh()
        )
        
        # Expert 3: 1200-2000Hz (Phase Precision) - Kernel 3 + LINEAR
        self.high_branch = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8), nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=1)
        )

    def forward(self, x):
        x_in = x.transpose(1, 2)
        # Weighting branches: sum equals 1.0, keeping output in linear range
        y_d = self.drone_branch(x_in) * 0.4
        y_m = self.mid_branch(x_in) * 0.3
        y_h = self.high_branch(x_in) * 0.3
        return (y_d + y_m + y_h).transpose(1, 2)

# --- 3. Surgical Loss (FP32 Stable) ---
class SpectralSurgicalLoss(nn.Module):
    def forward(self, y_pred, y_true):
        mse = nn.MSELoss()(y_pred, y_true)
        
        # Force FP32 for FFT to avoid ComplexHalf experimental warnings
        y_pred_f32 = y_pred.squeeze(-1).float()
        y_true_f32 = y_true.squeeze(-1).float()
        
        p_fft = torch.abs(torch.fft.rfft(y_pred_f32))
        t_fft = torch.abs(torch.fft.rfft(y_true_f32))
        
        # Phantom Penalty: Stop noise injection in silent bands
        silence_mask = (t_fft < 1e-3).float()
        phantom_penalty = torch.mean(torch.square(p_fft) * silence_mask) * 500.0
        
        return mse + phantom_penalty.to(mse.dtype)

# --- 4. Data Loader ---
def load_v10_data(data_dir, scenarios):
    X, y = [], []
    print(f"Loading {len(scenarios)} scenarios...")
    for scn_id in scenarios:
        prefix = os.path.join(data_dir, f"scn_{scn_id:03d}")
        if not os.path.exists(f"{prefix}_ref.npy"): continue
        
        ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
        
        f_ref = convolve(ref, hs, mode='same')
        # Normalization to 0.7 for safety
        f_ref = (f_ref / (np.max(np.abs(f_ref)) + 1e-7)) * 0.7
        mic = (mic / (np.max(np.abs(mic)) + 1e-7)) * 0.7
        
        for j in range(0, len(f_ref)-WINDOW_SIZE, 128):
            X.append(f_ref[j:j+WINDOW_SIZE])
            y.append(mic[j:j+WINDOW_SIZE])
            
    return torch.tensor(np.array(X)).unsqueeze(-1).float(), torch.tensor(np.array(y)).unsqueeze(-1).float()

# --- 5. Main Execution ---
if __name__ == "__main__":
    DATA_DIR = "driver_bulk_4spk_data"
    X_train, y_train = load_v10_data(DATA_DIR, range(45))
    dataset = TensorDataset(X_train, -y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    model = SpectralExpertTCN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = SpectralSurgicalLoss()

    print(f"--- Training v10 on GPU: {torch.cuda.get_device_name(0)} ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.5f}")

    # --- Visualization ---
    print("\nFinalizing Performance Report...")
    model.eval()
    with torch.no_grad():
        sample_in = X_train[0:1].to(device)
        sample_target = -y_train[0, :, 0].numpy()
        x_in = sample_in.transpose(1, 2)
        d_out = (model.drone_branch(x_in) * 0.4).cpu().numpy()[0, 0, :]
        m_out = (model.mid_branch(x_in) * 0.3).cpu().numpy()[0, 0, :]
        h_out = (model.high_branch(x_in) * 0.3).cpu().numpy()[0, 0, :]
        full_res = sample_target + (d_out + m_out + h_out)

    f, p_orig = welch(sample_target, FS, nperseg=256)
    _, p_resid = welch(full_res, FS, nperseg=256)

    plt.figure(figsize=(12, 6))
    plt.semilogy(f, p_orig, label="Original Noise", alpha=0.5)
    plt.semilogy(f, p_resid, label="v10 Residual", color='green', lw=2)
    plt.axvspan(1600, 2000, color='red', alpha=0.1, label='Surgical Zone')
    plt.title("v10 Final Spectral Performance")
    plt.legend(); plt.grid(True)
    plt.savefig("v10_final_diagnostic.png")

    torch.save(model.state_dict(), "anc_v10_spectral_experts.pth")
    print("All processes complete. Weights saved as 'anc_v10_spectral_experts.pth'.")