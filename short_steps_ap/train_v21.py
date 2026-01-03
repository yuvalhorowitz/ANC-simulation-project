import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import weight_norm
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS, BATCH_SIZE, EPOCHS = 8000, 32, 150
WINDOW_SIZE, LATENCY_COMP = 512, 7

# --- 1. Gated TCN Block (Ref: Paper Section D) ---
class GatedTCNBlock(nn.Module):
    def __init__(self, in_c, out_c, k, d, dropout=0.1):
        super(GatedTCNBlock, self).__init__()
        self.padding = (k - 1) * d
        self.conv_data = weight_norm(nn.Conv1d(in_c, out_c, k, padding=self.padding, dilation=d))
        self.conv_gate = weight_norm(nn.Conv1d(in_c, out_c, k, padding=self.padding, dilation=d))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None

    def forward(self, x):
        # Causal slicing: remove the padding added to the 'future'
        data = self.conv_data(x)[:, :, :-self.padding]
        gate = self.sigmoid(self.conv_gate(x)[:, :, :-self.padding])
        # Gating: Stage 1 Activation (Sigmoid) controls the flow
        out = self.dropout(data * gate)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

# --- 2. Gated Architecture ---
class GatedTCN_v20(nn.Module):
    def __init__(self):
        super(GatedTCN_v20, self).__init__()
        self.tcn = nn.Sequential(
            GatedTCNBlock(1, 32, k=7, d=1),
            GatedTCNBlock(32, 32, k=7, d=2),
            GatedTCNBlock(32, 32, k=7, d=4),
            GatedTCNBlock(32, 32, k=7, d=8)
        )
        self.linear_head = nn.Conv1d(32, 1, 1)
        self.raw_gain = nn.Parameter(torch.ones(1) * -0.5)

    def forward(self, x):
        x_in = x.transpose(1, 2)
        y = self.tcn(x_in)
        y = torch.tanh(self.linear_head(y)) # Stage 2 Activation: -1 to 1 range
        active_gain = 0.5 + 0.6 * torch.sigmoid(self.raw_gain)
        return (-1.0 * y * active_gain).transpose(1, 2)

# --- 3. Loss & Data Loading ---
class StrictGatedLoss(nn.Module):
    def forward(self, y_pred, y_true, epoch):
        mse = nn.MSELoss()(y_pred, -y_true)
        y_p, y_t = y_pred.squeeze(-1), y_true.squeeze(-1)
        corr = torch.mean(y_p * y_t) / (torch.std(y_p) * torch.std(y_t) + 1e-8)
        # Force negative correlation
        phase_penalty = torch.relu(corr + 0.7) * 150.0 
        weight = 1.0 if epoch < 60 else 15.0
        return (mse * weight) + phase_penalty

def load_data(data_dir, scenarios):
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
    X_train, y_train = load_data("driver_bulk_4spk_data", range(45))
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    model = GatedTCN_v20().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = StrictGatedLoss()
    history = []

    for epoch in range(EPOCHS):
        model.train()
        l_sum = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(); output = model(data); loss = criterion(output, target, epoch)
            loss.backward(); optimizer.step(); l_sum += loss.item()
        history.append(l_sum/len(loader))
        if (epoch+1)%10==0: print(f"Epoch {epoch+1} | Loss: {history[-1]:.5f}")

    # Plot Training Report
    model.eval()
    with torch.no_grad():
        s_in = X_train[0:1].to(device)
        s_target = y_train[0, :, 0].numpy()
        s_anti = model(s_in).cpu().numpy()[0, :, 0]
        s_res = s_target + s_anti
    
    plt.figure(figsize=(12, 6))
    plt.plot(s_target[:400], label="Noise", alpha=0.5)
    plt.plot(s_anti[:400], label="Anti-Noise (Inverted)", ls='--')
    plt.title("v20 Training Alignment Check"); plt.legend(); plt.savefig("v20_train_report.png")
    torch.save(model.state_dict(), "anc_v20_gated.pth")