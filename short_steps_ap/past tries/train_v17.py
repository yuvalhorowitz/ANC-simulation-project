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
FS, BATCH_SIZE, EPOCHS = 8000, 32, 150
WINDOW_SIZE, LATENCY_COMP = 512, 7 

# --- 2. Architecture: TCN with Internal Phase Inversion ---
class PhaseInvertedTCN_v17(nn.Module):
    def __init__(self):
        super(PhaseInvertedTCN_v17, self).__init__()
        # Drone Branch (50-400Hz) - Follows TCN 'Distinguishing Characteristics' [cite: 72, 73]
        self.drone_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, padding=15),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 32, kernel_size=7, padding=6, dilation=2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 32, kernel_size=7, padding=12, dilation=4), 
            nn.BatchNorm1d(32), nn.LeakyReLU(0.1),
            nn.Conv1d(32, 1, kernel_size=1), nn.Tanh()
        )
        # High Branch (1.2k-2k)
        self.high_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.LeakyReLU(0.1),
            nn.Conv1d(16, 1, kernel_size=1)
        )
        self.raw_gain = nn.Parameter(torch.ones(1) * 0.0)

    def forward(self, x):
        # Maps input sequence to output sequence of same length [cite: 74, 92]
        x_in = x.transpose(1, 2)
        y_d = self.drone_branch(x_in) * 0.85
        y_h = self.high_branch(x_in) * 0.15
        
        # Gain mapped to [0.5, 1.5] to ensure aggressive cancellation
        active_gain = 0.5 + 1.0 * torch.sigmoid(self.raw_gain)
        
        # THE CORE FIX: INTERNAL INVERSION
        # We output the negative signal directly to enforce anti-phase behavior
        anti_noise = (y_d + y_h) * active_gain
        return (-1.0 * anti_noise).transpose(1, 2)

# --- 3. Phase-Strict Loss ---
class PhaseStrictLoss(nn.Module):
    def forward(self, y_pred, y_true, epoch):
        # Since model outputs negative, we want y_pred + y_true to be 0
        mse = nn.MSELoss()(y_pred, -y_true)
        
        y_p, y_t = y_pred.squeeze(-1), y_true.squeeze(-1)
        # We want negative correlation (destructive interference)
        correlation = torch.mean(y_p * y_t)
        phase_penalty = torch.relu(correlation + 0.1) * 50.0 # Heavy penalty for in-phase
        
        # Energy weight activates after phase is locked
        energy_gap = torch.abs(torch.std(y_p) - torch.std(y_t))
        energy_weight = 0.0 if epoch < 60 else 15.0
        
        mse_weight = 1.0 if epoch < 60 else 10.0
        return (mse * mse_weight) + phase_penalty + (energy_gap * energy_weight)

# --- 4. Load Data ---
def load_v17_data(data_dir, scenarios):
    X, y = [], []
    for scn_id in scenarios:
        prefix = os.path.join(data_dir, f"scn_{scn_id:03d}")
        if not os.path.exists(f"{prefix}_ref.npy"): continue
        ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
        f_ref = convolve(ref, hs, mode='same')
        f_ref = (f_ref / (np.max(np.abs(f_ref)) + 1e-7)) * 0.7
        mic = (mic / (np.max(np.abs(mic)) + 1e-7)) * 0.7
        for j in range(0, len(f_ref) - WINDOW_SIZE - LATENCY_COMP, 128):
            X.append(f_ref[j : j+WINDOW_SIZE])
            y.append(mic[j + LATENCY_COMP : j + WINDOW_SIZE + LATENCY_COMP])
    return torch.from_numpy(np.array(X)).unsqueeze(-1).float(), \
           torch.from_numpy(np.array(y)).unsqueeze(-1).float()

if __name__ == "__main__":
    # CRITICAL: We train against the positive signal because model inverts internally
    X_train, y_train = load_v17_data("driver_bulk_4spk_data", range(45))
    dataset = TensorDataset(X_train, y_train) 
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PhaseInvertedTCN_v17().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = PhaseStrictLoss()
    history = []

    print(f"--- Training v17 Phase-Inverted on {device} ---")
    for epoch in range(EPOCHS):
        model.train()
        l_sum = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target, epoch)
            loss.backward()
            optimizer.step()
            l_sum += loss.item()
        
        history.append(l_sum/len(loader))
        current_gain = 0.5 + 1.0 * torch.sigmoid(model.raw_gain).item()
        if (epoch+1)%10 == 0:
            print(f"Epoch {epoch+1} | Loss: {history[-1]:.5f} | Gain: {current_gain:.3f}")

    # Visualization
    model.eval()
    with torch.no_grad():
        sample_in = X_train[0:1].to(device)
        sample_target = y_train[0, :, 0].numpy()
        anti_noise = model(sample_in).cpu().numpy()[0, :, 0]
        combined = sample_target + anti_noise # Destructive because anti_noise is negative

    fig, axs = plt.subplots(4, 1, figsize=(15, 20))
    axs[0].plot(sample_target[:600], label="Original Noise", alpha=0.3)
    axs[0].plot(anti_noise[:600], label="v17 Anti-Noise (Inverted)", color='red', ls='--')
    axs[0].plot(combined[:600], label="Residual", color='green', lw=2)
    axs[0].set_title("Time Domain phase Alignment Check"); axs[0].legend()
    
    f, p_orig = welch(sample_target, FS, nperseg=256)
    _, p_resid = welch(combined, FS, nperseg=256)
    axs[2].semilogy(f, p_orig, label="Original"); axs[2].semilogy(f, p_resid, color='green', label="Residual")
    axs[2].fill_between(f, p_resid, p_orig, where=(p_resid < p_orig), color='green', alpha=0.2)
    axs[2].set_title("Spectral Suppression Report"); axs[2].legend()

    axs[3].plot(history); axs[3].set_title("Loss History")
    plt.tight_layout(); plt.savefig("v17_phase_inverted_report.png")
    torch.save(model.state_dict(), "anc_v17_phase_inverted.pth")