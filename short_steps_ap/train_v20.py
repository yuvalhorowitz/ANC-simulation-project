import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import weight_norm
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

# --- 1. Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS, BATCH_SIZE, EPOCHS = 8000, 32, 150
WINDOW_SIZE, LATENCY_COMP = 512, 7 

# --- 2. Gated Residual Block (Ref: Paper Section D) ---
class GatedTCNBlock_v20(nn.Module):
    def __init__(self, in_c, out_c, k, d, dropout=0.1):
        super(GatedTCNBlock_v20, self).__init__()
        # Strict causality: Padding only on the 'past' side
        self.padding = (k - 1) * d
        
        # TCN Paper Section D: Product between two layers (Data & Gate)
        # Note: Gated architecture uses approx. 2x the layers [cite: 528]
        self.conv_data = weight_norm(nn.Conv1d(in_c, out_c, k, padding=self.padding, dilation=d))
        self.conv_gate = weight_norm(nn.Conv1d(in_c, out_c, k, padding=self.padding, dilation=d))
        
        self.sigmoid = nn.Sigmoid() # The Gate
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None

    def forward(self, x):
        # Slice the 'future' side out to maintain strict causality 
        data = self.conv_data(x)[:, :, :-self.padding]
        gate = self.sigmoid(self.conv_gate(x)[:, :, :-self.padding])
        
        # Gating operation: allows model to block out destructive phase 
        out = self.dropout(data * gate)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

# --- 3. Gated Architecture v20 ---
class GatedTCN_v20(nn.Module):
    def __init__(self):
        super(GatedTCN_v20, self).__init__()
        # Exponential dilations enable a massive receptive field [cite: 103, 146]
        self.tcn = nn.Sequential(
            GatedTCNBlock_v20(1, 32, k=7, d=1),
            GatedTCNBlock_v20(32, 32, k=7, d=2),
            GatedTCNBlock_v20(32, 32, k=7, d=4),
            GatedTCNBlock_v20(32, 32, k=7, d=8)
        )
        self.linear_head = nn.Conv1d(32, 1, 1)
        # Initialize gain lower for stability
        self.raw_gain = nn.Parameter(torch.ones(1) * -0.5)

    def forward(self, x):
        x_in = x.transpose(1, 2)
        y = self.tcn(x_in)
        y = self.linear_head(y)
        
        # Gain authority capped at 1.1x to prevent constructive runaway
        active_gain = 0.5 + 0.6 * torch.sigmoid(self.raw_gain)
        
        # Internal Phase Inversion (180 deg shift)
        return (-1.0 * y * active_gain).transpose(1, 2)

# --- 4. Causal Trainer & Loss ---
class CorrelationGuardLoss(nn.Module):
    def forward(self, y_pred, y_true, epoch):
        mse = nn.MSELoss()(y_pred, -y_true)
        y_p, y_t = y_pred.squeeze(-1), y_true.squeeze(-1)
        
        # Correlation Guard: We want -1.0
        # Punish if correlation is > -0.7 (forcing destructive interference)
        corr = torch.mean(y_p * y_t) / (torch.std(y_p) * torch.std(y_t) + 1e-8)
        phase_guard = torch.relu(corr + 0.7) * 150.0 # Aggressive Phase Guard
        
        weight = 1.0 if epoch < 60 else 15.0
        return (mse * weight) + phase_guard

def load_v20_data(data_dir, scenarios):
    X, y = [], []
    for scn_id in scenarios:
        prefix = os.path.join(data_dir, f"scn_{scn_id:03d}")
        if not os.path.exists(f"{prefix}_ref.npy"): continue
        ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
        f_ref = (convolve(ref, hs, mode='same') / (np.max(np.abs(ref)) + 1e-7)) * 0.7
        mic = (mic / (np.max(np.abs(mic)) + 1e-7)) * 0.7
        for j in range(0, len(f_ref) - WINDOW_SIZE - LATENCY_COMP, 128):
            X.append(f_ref[j : j+WINDOW_SIZE])
            y.append(mic[j + LATENCY_COMP : j + WINDOW_SIZE + LATENCY_COMP])
    return torch.from_numpy(np.array(X)).unsqueeze(-1).float(), \
           torch.from_numpy(np.array(y)).unsqueeze(-1).float()

if __name__ == "__main__":
    X_train, y_train = load_v20_data("driver_bulk_4spk_data", range(45))
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GatedTCN_v20().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = CorrelationGuardLoss()
    history = []

    print(f"--- Training v20 Gated TCN on {device} ---")
    for epoch in range(150):
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
        if (epoch+1)%10 == 0:
            g = 0.5 + 0.6 * torch.sigmoid(model.raw_gain).item()
            print(f"Epoch {epoch+1} | Loss: {history[-1]:.5f} | Gain: {g:.3f}")

    # Plotting Logic
    model.eval()
    with torch.no_grad():
        sample_in = X_train[0:1].to(device)
        sample_target = y_train[0, :, 0].numpy()
        anti_noise = model(sample_in).cpu().numpy()[0, :, 0]
        combined = sample_target + anti_noise

    fig, axs = plt.subplots(4, 1, figsize=(15, 20))
    axs[0].plot(sample_target[:600], label="Noise", alpha=0.3)
    axs[0].plot(anti_noise[:600], label="v20 Gated Anti-Noise", color='red', ls='--')
    axs[0].plot(combined[:600], label="Residual", color='green', lw=2)
    axs[0].set_title("1. Gated Phase Alignment (Destructive Proof)"); axs[0].legend()
    
    f, p_orig = welch(sample_target, 8000, nperseg=256)
    _, p_resid = welch(combined, 8000, nperseg=256)
    axs[2].semilogy(f, p_orig, label="Original"); axs[2].semilogy(f, p_resid, color='green', label="Residual")
    axs[2].fill_between(f, p_resid, p_orig, where=(p_resid < p_orig), color='green', alpha=0.2)
    axs[2].set_title("3. Spectral Suppression Report"); axs[2].legend()
    
    axs[3].plot(history); axs[3].set_title("4. Loss History (Gating Active)")
    plt.tight_layout(); plt.savefig("v20_gated_report.png")
    torch.save(model.state_dict(), "anc_v20_gated.pth")