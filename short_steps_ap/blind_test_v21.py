import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = 8000

# (Keep Model Definitions Identity Match to Trainer)
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
        d = self.conv_data(x)[:, :, :-self.padding]; g = self.sigmoid(self.conv_gate(x)[:, :, :-self.padding])
        return self.dropout(d * g) + (x if self.downsample is None else self.downsample(x))

class GatedTCN_v20(nn.Module):
    def __init__(self):
        super(GatedTCN_v20, self).__init__()
        self.tcn = nn.Sequential(GatedTCNBlock(1, 32, 7, 1), GatedTCNBlock(32, 32, 7, 2), 
                                 GatedTCNBlock(32, 32, 7, 4), GatedTCNBlock(32, 32, 7, 8))
        self.linear_head = nn.Conv1d(32, 1, 1); self.raw_gain = nn.Parameter(torch.ones(1) * -0.5)
    def forward(self, x):
        y = torch.tanh(self.linear_head(self.tcn(x.transpose(1, 2))))
        return (-1.0 * y * (0.5 + 0.6 * torch.sigmoid(self.raw_gain))).transpose(1, 2)

def run_test():
    model = GatedTCN_v20().to(device)
    model.load_state_dict(torch.load("anc_v20_gated.pth", map_location=device))
    model.eval()

    results = []
    plot_data = None

    with torch.no_grad():
        for scn_id in range(45, 50):
            prefix = f"driver_bulk_4spk_data/scn_{scn_id:03d}"
            if not os.path.exists(f"{prefix}_ref.npy"): continue
            ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
            f_ref = (convolve(ref, hs, mode='same') / (np.max(np.abs(ref)) + 1e-7)) * 0.7
            mic = (mic / (np.max(np.abs(mic)) + 1e-7)) * 0.7
            X = torch.from_numpy(np.array([f_ref[i:i+512] for i in range(0, len(f_ref)-512, 128)])).unsqueeze(-1).float().to(device)
            u = model(X).cpu().detach().numpy()
            rec, gen = mic[512+7:], u[:, -1, 0]; min_l = min(len(rec), len(gen))
            res = rec[:min_l] + gen[:min_l]
            db = 10 * np.log10(np.mean(res**2) / (np.mean(rec[:min_l]**2) + 1e-12))
            results.append((scn_id, db))
            if plot_data is None: plot_data = (rec[:min_l], gen[:min_l], res, scn_id)

    # --- Plotting Diagnostic PNG ---
    rec, gen, res, s_id = plot_data
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    
    # 1. Full Time Domain (Destructive Check)
    axs[0].plot(rec[:600], label="Noise (Original)", color='blue', alpha=0.4)
    axs[0].plot(gen[:600], label="Anti-Noise (Inverted)", color='red', ls='--')
    axs[0].plot(res[:600], label="Residual (Error)", color='green', lw=2)
    axs[0].set_title(f"Time Domain Analysis (Scenario {s_id})"); axs[0].legend(); axs[0].grid(True)

    # 2. Frequency Domain (Spectrum Check)
    f, p_orig = welch(rec, FS, nperseg=512); _, p_res = welch(res, FS, nperseg=512)
    axs[1].semilogy(f, p_orig, label="Original PSD"); axs[1].semilogy(f, p_res, label="Cancelled PSD", color='green')
    axs[1].fill_between(f, p_res, p_orig, where=(p_res < p_orig), color='green', alpha=0.2)
    axs[1].set_title("Power Spectral Density (0-2000Hz)"); axs[1].set_xlim(0, 2000); axs[1].legend(); axs[1].grid(True)

    # 3. Final Performance Table
    cell_text = [[f"Scenario {r[0]}", f"{r[1]:.2f} dB"] for r in results]
    axs[2].axis('off'); axs[2].table(cellText=cell_text, colLabels=["Scenario", "System Gain"], loc='center').scale(0.8, 2)
    
    plt.tight_layout(); plt.savefig("v20_blind_test_report.png")
    print("Complete PNG reports saved.")

if __name__ == "__main__":
    run_test()