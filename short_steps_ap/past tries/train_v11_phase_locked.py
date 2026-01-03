import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

# --- 1. GPU & Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS, BATCH_SIZE, EPOCHS = 8000, 32, 100
WINDOW_SIZE = 512
scaler = torch.amp.GradScaler('cuda')

# --- 2. Model Architecture: Dilated Spectral Experts ---
class SpectralExpertTCN_v11(nn.Module):
    def __init__(self):
        super(SpectralExpertTCN_v11, self).__init__()
        
        # Expert 1: 50-400Hz (Drone) - Receptive Field: 512 samples via Dilation
        self.drone_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1, dilation=1), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=8, dilation=8), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=16, dilation=16), 
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1), nn.Tanh()
        )
        
        # Expert 2: 400-1200Hz (Mid) - Receptive Field: 128 samples
        self.mid_branch = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=3, padding=1, dilation=1), nn.ReLU(),
            nn.Conv1d(12, 12, kernel_size=3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(12, 12, kernel_size=3, padding=4, dilation=4), 
            nn.BatchNorm1d(12), nn.ReLU(),
            nn.Conv1d(12, 1, kernel_size=1), nn.Tanh()
        )
        
        # Expert 3: 1200-2000Hz (High) - Fast Response, Linear
        self.high_branch = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8), nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=1)
        )

    def forward(self, x):
        x_in = x.transpose(1, 2)
        # Power Shift: 60% Drone authority for deeper engine cancellation
        y_d = self.drone_branch(x_in) * 0.6
        y_m = self.mid_branch(x_in) * 0.25
        y_h = self.high_branch(x_in) * 0.15
        return (y_d + y_m + y_h).transpose(1, 2)

# --- 3. Loss Function ---
class SurgicalPhaseLoss(nn.Module):
    def forward(self, y_pred, y_true):
        mse = nn.MSELoss()(y_pred, y_true)
        # FP32 Spectral Protection
        y_p_f32 = y_pred.squeeze(-1).float()
        y_t_f32 = y_true.squeeze(-1).float()
        p_fft = torch.abs(torch.fft.rfft(y_p_f32))
        t_fft = torch.abs(torch.fft.rfft(y_t_f32))
        phantom_penalty = torch.mean(torch.square(p_fft) * (t_fft < 1e-3).float()) * 500.0
        return mse + phantom_penalty.to(mse.dtype)

# --- 4. Data Loading ---
def load_v11_data(data_dir, scenarios):
    X, y = [], []
    print(f"Loading {len(scenarios)} scenarios...")
    for scn_id in scenarios:
        prefix = os.path.join(data_dir, f"scn_{scn_id:03d}")
        if not os.path.exists(f"{prefix}_ref.npy"): continue
        ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
        f_ref = convolve(ref, hs, mode='same')
        f_ref = (f_ref / (np.max(np.abs(f_ref)) + 1e-7)) * 0.7
        mic = (mic / (np.max(np.abs(mic)) + 1e-7)) * 0.7
        for j in range(0, len(f_ref)-WINDOW_SIZE, 128):
            X.append(f_ref[j:j+WINDOW_SIZE]); y.append(mic[j:j+WINDOW_SIZE])
    return torch.tensor(np.array(X)).unsqueeze(-1).float(), torch.tensor(np.array(y)).unsqueeze(-1).float()

# --- 5. Main Loop ---
if __name__ == "__main__":
    DATA_DIR = "driver_bulk_4spk_data"
    X_train, y_train = load_v11_data(DATA_DIR, range(45))
    dataset = TensorDataset(X_train, -y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SpectralExpertTCN_v11().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = SurgicalPhaseLoss()
    history = []

    print(f"--- Training v11 Phase-Locked on {device} ---")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        history.append(epoch_loss/len(loader))
        if (epoch+1)%10 == 0: print(f"Epoch {epoch+1} | Loss: {history[-1]:.5f}")

    # --- 6. Detailed 4-Panel Visualization ---
    print("\nGenerating v11 Performance Report...")
    model.eval()
    with torch.no_grad():
        sample_in = X_train[0:1].to(device)
        sample_target = -y_train[0, :, 0].numpy()
        anti_phase = model(sample_in).cpu().numpy()[0, :, 0]
        combined = sample_target + anti_phase

    fig, axs = plt.subplots(4, 1, figsize=(15, 20))

    # Plot 1: Time Domain Analysis
    axs[0].plot(sample_target[:600], label="Original Noise", color='blue', alpha=0.3)
    axs[0].plot(anti_phase[:600], label="v11 Anti-Phase", color='red', linestyle='--')
    axs[0].plot(combined[:600], label="Result (Residual)", color='green', lw=2)
    axs[0].set_title("1. Time Domain: Phase Inversion Alignment"); axs[0].legend(); axs[0].grid(True)

    # Plot 2: Harmonics/Whistle Check (Frequency Plot)
    f, p_gen = welch(anti_phase, FS, nperseg=256)
    axs[1].plot(f, 10*np.log10(p_gen + 1e-12), color='orange', label="Generated Anti-Noise Spectrum")
    axs[1].axvspan(1600, 2000, color='red', alpha=0.1, label='Surgical Zone')
    axs[1].set_title("2. Harmonic Analysis: Checking for Self-Generated Whistles"); axs[1].set_xlim(0, 2200); axs[1].legend()

    # Plot 3: PSD Reduction Map
    f, p_orig = welch(sample_target, FS, nperseg=256)
    _, p_resid = welch(combined, FS, nperseg=256)
    axs[2].semilogy(f, p_orig, label="Original Noise", alpha=0.4)
    axs[2].semilogy(f, p_resid, label="v11 Residual", color='green', lw=2)
    axs[2].fill_between(f, p_resid, p_orig, where=(p_resid < p_orig), color='green', alpha=0.2, label="Reduction")
    axs[2].set_title("3. Combined PSD: Sound Reduction Map"); axs[2].set_xlim(0, 2200); axs[2].legend()

    # Plot 4: Training Convergence
    axs[3].plot(history, color='black', lw=2)
    axs[3].set_title("4. Loss Level Across Epochs"); axs[3].set_xlabel("Epoch"); axs[3].grid(True)

    plt.tight_layout()
    plt.savefig("v11_detailed_training_report.png")
    torch.save(model.state_dict(), "anc_v11_phase_locked.pth")
    print("Report saved as 'v11_detailed_training_report.png'. Model ready for Blind Test.")