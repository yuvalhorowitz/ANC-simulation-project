import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

# --- 1. Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS, BATCH_SIZE, EPOCHS = 8000, 32, 120
WINDOW_SIZE = 512
# LATENCY_COMP: Compensates for speaker-to-mic distance (~0.3m = 7 samples)
LATENCY_COMP = 7 

# --- 2. Predictive Architecture ---
class PredictiveExpertTCN_v13(nn.Module):
    def __init__(self):
        super(PredictiveExpertTCN_v13, self).__init__()
        
        # Drone Branch (50-400Hz) - Aggressive Capacity
        self.drone_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, padding=15),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 32, kernel_size=7, padding=6, dilation=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 1, kernel_size=1), nn.Tanh()
        )
        
        # High-Freq Support (1.2k-2k) - Strictly Linear
        self.high_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1)
        )
        
        # Starting with aggressive gain to push past 0dB
        self.global_gain = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, x):
        x_in = x.transpose(1, 2)
        # 85% Authority to engine drone
        y_d = self.drone_branch(x_in) * 0.85
        y_h = self.high_branch(x_in) * 0.15
        return ((y_d + y_h) * self.global_gain).transpose(1, 2)

# --- 3. Predictive Data Loading ---
def load_v13_predictive(data_dir, scenarios):
    X, y = [], []
    print(f"Loading {len(scenarios)} scenarios with {LATENCY_COMP} sample look-ahead...")
    for scn_id in scenarios:
        prefix = os.path.join(data_dir, f"scn_{scn_id:03d}")
        if not os.path.exists(f"{prefix}_ref.npy"): continue
        ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
        
        f_ref = convolve(ref, hs, mode='same')
        f_ref = (f_ref / (np.max(np.abs(f_ref)) + 1e-7)) * 0.7
        mic = (mic / (np.max(np.abs(mic)) + 1e-7)) * 0.7
        
        # --- THE ALIGNMENT FIX ---
        # Shift target 'y' back to force the model to lead the phase
        for j in range(0, len(f_ref) - WINDOW_SIZE - LATENCY_COMP, 128):
            X.append(f_ref[j : j+WINDOW_SIZE])
            y.append(mic[j + LATENCY_COMP : j + WINDOW_SIZE + LATENCY_COMP])
            
    return torch.tensor(np.array(X)).unsqueeze(-1).float(), torch.tensor(np.array(y)).unsqueeze(-1).float()

# --- 4. Main Training Loop ---
if __name__ == "__main__":
    DATA_DIR = "driver_bulk_4spk_data"
    X_train, y_train = load_v13_predictive(DATA_DIR, range(45))
    dataset = TensorDataset(X_train, -y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PredictiveExpertTCN_v13().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    history = []

    print(f"--- Training v13 Predictive on {device} ---")
    for epoch in range(EPOCHS):
        model.train()
        l_sum = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            l_sum += loss.item()
        
        history.append(l_sum/len(loader))
        if (epoch+1)%10 == 0:
            print(f"Epoch {epoch+1} | Loss: {history[-1]:.6f} | Gain: {model.global_gain.item():.2f}")

    # --- 5. Visualization Report ---
    print("\nGenerating Predictive Performance Report...")
    model.eval()
    with torch.no_grad():
        sample_in = X_train[0:1].to(device)
        sample_target = -y_train[0, :, 0].numpy()
        anti_phase = model(sample_in).cpu().numpy()[0, :, 0]
        combined = sample_target + anti_phase

    fig, axs = plt.subplots(4, 1, figsize=(15, 20))

    # Time Domain (Checking Phase Advancement)
    axs[0].plot(sample_target[:600], label="Original Noise", color='blue', alpha=0.3)
    axs[0].plot(anti_phase[:600], label="v13 Predictive Anti-Noise", color='red', linestyle='--')
    axs[0].plot(combined[:600], label="Residual", color='green', lw=2)
    axs[0].set_title(f"1. Predictive Phase Alignment (Look-ahead: {LATENCY_COMP} samples)"); axs[0].legend(); axs[0].grid(True)

    # Generated Spectrum (Harmonic Check)
    f, p_gen = welch(anti_phase, FS, nperseg=256)
    axs[1].plot(f, 10*np.log10(p_gen + 1e-12), color='orange', label="Anti-Noise Spectrum")
    axs[1].set_title("2. Harmonic Injection Analysis"); axs[1].set_xlim(0, 2200); axs[1].legend()

    # PSD Reduction
    f, p_orig = welch(sample_target, FS, nperseg=256)
    _, p_resid = welch(combined, FS, nperseg=256)
    axs[2].semilogy(f, p_orig, label="Original Noise", alpha=0.4)
    axs[2].semilogy(f, p_resid, label="v13 Residual", color='green', lw=2)
    axs[2].fill_between(f, p_resid, p_orig, where=(p_resid < p_orig), color='green', alpha=0.2)
    axs[2].set_title("3. Spectral Power Reduction (Target: Positive dB)"); axs[2].set_xlim(0, 2200); axs[2].legend()

    # Loss
    axs[3].plot(history, color='black', lw=2)
    axs[3].set_title("4. Convergence History"); axs[3].set_xlabel("Epoch"); axs[3].grid(True)

    plt.tight_layout()
    plt.savefig("v13_predictive_report.png")
    torch.save(model.state_dict(), "anc_v13_predictive.pth")
    print("Success. Saved as 'anc_v13_predictive.pth'.")