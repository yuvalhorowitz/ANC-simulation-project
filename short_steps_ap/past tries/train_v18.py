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

# --- 2. TCN Residual Block (As per Figure 1(b) in paper) ---
class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNResidualBlock, self).__init__()
        # Each block consists of two layers of dilated causal convolution and non-linearity [cite: 159]
        # Padding is set to (k-1)*d to ensure causal output length [cite: 92]
        padding = (kernel_size - 1) * dilation
        
        # Layer 1
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, 
                                           padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU() # [cite: 159]
        self.dropout1 = nn.Dropout(dropout) # [cite: 161]

        # Layer 2
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, 
                                           padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        
        # 1x1 convolution is added when residual input and output have different dimensions [cite: 140, 163]
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_final = nn.ReLU() # o = Activation(x + F(x)) [cite: 151]

    def forward(self, x):
        out = self.net(x)
        # We must slice the output of the causal convolutions to match length [cite: 92]
        out = out[:, :, :x.size(2)] 
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_final(out + res)

# --- 3. Final Architecture v18 ---
class FinalTCN_v18(nn.Module):
    def __init__(self):
        super(FinalTCN_v18, self).__init__()
        # Exponential dilations enable an exponentially large receptive field 
        self.tcn = nn.Sequential(
            TCNResidualBlock(1, 32, kernel_size=7, dilation=1),
            TCNResidualBlock(32, 32, kernel_size=7, dilation=2),
            TCNResidualBlock(32, 32, kernel_size=7, dilation=4),
            TCNResidualBlock(32, 32, kernel_size=7, dilation=8),
            TCNResidualBlock(32, 32, kernel_size=7, dilation=16)
        )
        self.linear = nn.Conv1d(32, 1, 1)
        self.raw_gain = nn.Parameter(torch.ones(1) * 0.0)

    def forward(self, x):
        x_in = x.transpose(1, 2)
        y = self.tcn(x_in)
        y = self.linear(y)
        
        active_gain = 0.5 + 1.0 * torch.sigmoid(self.raw_gain)
        # Internal Inversion to ensure destructive interference
        anti_noise = y * active_gain
        return (-1.0 * anti_noise).transpose(1, 2)

# --- 4. Loss & Loading Logic ---
class PhaseStrictLoss(nn.Module):
    def forward(self, y_pred, y_true, epoch):
        mse = nn.MSELoss()(y_pred, -y_true)
        y_p, y_t = y_pred.squeeze(-1), y_true.squeeze(-1)
        correlation = torch.mean(y_p * y_t)
        # Punish any in-phase behavior [Correlation > -0.1]
        phase_penalty = torch.relu(correlation + 0.1) * 50.0 
        
        weight = 1.0 if epoch < 60 else 10.0
        return (mse * weight) + phase_penalty

def load_v18_data(data_dir, scenarios):
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
    X_train, y_train = load_v18_data("driver_bulk_4spk_data", range(45))
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = FinalTCN_v18().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = PhaseStrictLoss()
    history = []

    print(f"--- Training v18 Residual TCN on {device} ---")
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

    # Plotting
    model.eval()
    with torch.no_grad():
        sample_in = X_train[0:1].to(device)
        sample_target = y_train[0, :, 0].numpy()
        anti_noise = model(sample_in).cpu().numpy()[0, :, 0]
        combined = sample_target + anti_noise

    fig, axs = plt.subplots(4, 1, figsize=(15, 20))
    axs[0].plot(sample_target[:600], label="Noise", alpha=0.3)
    axs[0].plot(anti_noise[:600], label="Anti-Noise (Inverted)", color='red', ls='--')
    axs[0].plot(combined[:600], label="Residual", color='green', lw=2)
    axs[0].set_title("1. Time Domain Phase Lock Analysis"); axs[0].legend()
    
    f, p_orig = welch(sample_target, FS, nperseg=256)
    _, p_resid = welch(combined, FS, nperseg=256)
    axs[2].semilogy(f, p_orig, label="Original"); axs[2].semilogy(f, p_resid, color='green', label="Residual")
    axs[2].fill_between(f, p_resid, p_orig, where=(p_resid < p_orig), color='green', alpha=0.2)
    axs[2].set_title("3. Spectral Suppression Report (v18 Residual)"); axs[2].legend()

    axs[3].plot(history); axs[3].set_title("4. Loss Convergence")
    plt.tight_layout(); plt.savefig("v18_residual_report.png")
    torch.save(model.state_dict(), "anc_v18_residual.pth")