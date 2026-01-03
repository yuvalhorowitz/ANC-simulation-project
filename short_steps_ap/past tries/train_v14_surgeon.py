import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

# --- 1. Config & GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS, BATCH_SIZE, EPOCHS = 8000, 32, 120
WINDOW_SIZE = 512
# Set this to the physical delay of your car cabin (try 7 or 8)
LATENCY_COMP = 7 

# --- 2. Surgeon Architecture ---
class PhaseSurgeonTCN_v14(nn.Module):
    def __init__(self):
        super(PhaseSurgeonTCN_v14, self).__init__()
        # Drone Branch (50-400Hz) - High Capacity
        self.drone_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, padding=15),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 32, kernel_size=7, padding=6, dilation=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 1, kernel_size=1), nn.Tanh()
        )
        # Linear High Branch
        self.high_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.LeakyReLU(0.1),
            nn.Conv1d(16, 1, kernel_size=1)
        )
        # Starting with a safer gain to avoid initial instability
        self.global_gain = nn.Parameter(torch.ones(1) * 0.9)

    def forward(self, x):
        x_in = x.transpose(1, 2)
        y_d = self.drone_branch(x_in) * 0.80
        y_h = self.high_branch(x_in) * 0.20
        return ((y_d + y_h) * self.global_gain).transpose(1, 2)

# --- 3. Phase-Enforced Loss ---
class PhaseEnforcedLoss(nn.Module):
    def forward(self, y_pred, y_true):
        # Time Domain Error
        mse = nn.MSELoss()(y_pred, y_true)
        
        # Correlation Penalty: Force Destructive Interference
        # We want y_pred and y_true to be negatively correlated
        y_p = y_pred.squeeze(-1)
        y_t = y_true.squeeze(-1)
        
        # If mean(y_p * y_t) is positive, it means they are IN-PHASE (Adding noise)
        correlation = torch.mean(y_p * y_t)
        phase_penalty = torch.relu(correlation) * 20.0 # Heavy penalty for in-phase
        
        # Spectral Whistle Guard (FP32)
        y_p_f32, y_t_f32 = y_p.float(), y_t.float()
        p_fft = torch.abs(torch.fft.rfft(y_p_f32))
        t_fft = torch.abs(torch.fft.rfft(y_t_f32))
        phantom_penalty = torch.mean(torch.square(p_fft) * (t_fft < 1e-3).float()) * 500.0
        
        return mse + phase_penalty + phantom_penalty.to(mse.dtype)

# --- 4. Data Loading ---
def load_v14_data(data_dir, scenarios):
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
    return torch.tensor(np.array(X)).unsqueeze(-1).float(), torch.tensor(np.array(y)).unsqueeze(-1).float()

# --- 5. Main Loop ---
if __name__ == "__main__":
    X_train, y_train = load_v14_data("driver_bulk_4spk_data", range(45))
    dataset = TensorDataset(X_train, -y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PhaseSurgeonTCN_v14().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = PhaseEnforcedLoss()
    history = []

    print(f"--- Training v14 Surgeon on {device} ---")
    for epoch in range(EPOCHS):
        model.train()
        l_sum = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            l_sum += loss.item()
        history.append(l_sum/len(loader))
        if (epoch+1)%10 == 0:
            print(f"Epoch {epoch+1} | Loss: {history[-1]:.6f} | Gain: {model.global_gain.item():.2f}")

    # --- 6. Visualization ---
    model.eval()
    with torch.no_grad():
        sample_in = X_train[0:1].to(device)
        sample_target = -y_train[0, :, 0].numpy()
        anti_phase = model(sample_in).cpu().numpy()[0, :, 0]
        combined = sample_target + anti_phase

    fig, axs = plt.subplots(4, 1, figsize=(15, 20))
    axs[0].plot(sample_target[:600], label="Original Noise", alpha=0.3)
    axs[0].plot(anti_phase[:600], label="v14 Anti-Phase", color='red', ls='--')
    axs[0].plot(combined[:600], label="Residual", color='green')
    axs[0].set_title(f"Time Domain Alignment (Gain: {model.global_gain.item():.2f})"); axs[0].legend()

    f, p_gen = welch(anti_phase, FS, nperseg=256)
    axs[1].plot(f, 10*np.log10(p_gen + 1e-12), color='orange')
    axs[1].set_title("Harmonic Generation Check"); axs[1].set_xlim(0, 2200)

    f, p_orig = welch(sample_target, FS, nperseg=256)
    _, p_resid = welch(combined, FS, nperseg=256)
    axs[2].semilogy(f, p_orig, label="Original"); axs[2].semilogy(f, p_resid, label="Residual", color='green')
    axs[2].fill_between(f, p_resid, p_orig, where=(p_resid < p_orig), color='green', alpha=0.2)
    axs[2].set_title("PSD Reduction Map"); axs[2].set_xlim(0, 2200); axs[2].legend()

    axs[3].plot(history); axs[3].set_title("Convergence History")
    plt.tight_layout(); plt.savefig("v14_surgeon_report.png")
    torch.save(model.state_dict(), "anc_v14_surgeon.pth")