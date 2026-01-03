import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = 8000

class GatedTCNBlock_v20(nn.Module):
    def __init__(self, in_c, out_c, k, d, dropout=0.1):
        super(GatedTCNBlock_v20, self).__init__()
        self.padding = (k - 1) * d
        self.conv_data = weight_norm(nn.Conv1d(in_c, out_c, k, padding=self.padding, dilation=d))
        self.conv_gate = weight_norm(nn.Conv1d(in_c, out_c, k, padding=self.padding, dilation=d))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else None
    def forward(self, x):
        data = self.conv_data(x)[:, :, :-self.padding]
        gate = self.sigmoid(self.conv_gate(x)[:, :, :-self.padding])
        out = self.dropout(data * gate)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class GatedTCN_v20(nn.Module):
    def __init__(self):
        super(GatedTCN_v20, self).__init__()
        self.tcn = nn.Sequential(
            GatedTCNBlock_v20(1, 32, k=7, d=1),
            GatedTCNBlock_v20(32, 32, k=7, d=2),
            GatedTCNBlock_v20(32, 32, k=7, d=4),
            GatedTCNBlock_v20(32, 32, k=7, d=8)
        )
        self.linear_head = nn.Conv1d(32, 1, 1)
        self.raw_gain = nn.Parameter(torch.ones(1) * -0.5)
    def forward(self, x):
        x_in = x.transpose(1, 2)
        y = self.tcn(x_in)
        y = self.linear_head(y)
        active_gain = 0.5 + 0.6 * torch.sigmoid(self.raw_gain)
        return (-1.0 * y * active_gain).transpose(1, 2)

def run_v20_test():
    model = GatedTCN_v20().to(device)
    model.load_state_dict(torch.load("anc_v20_gated.pth", map_location=device))
    model.eval()

    print("--- v20 Gated TCN Blind Test Diagnostic ---")
    all_metrics = []
    
    # We will pick one scenario to plot in detail (usually the best or first)
    plot_data = None

    with torch.no_grad():
        for scn_id in range(45, 50):
            prefix = f"driver_bulk_4spk_data/scn_{scn_id:03d}"
            if not os.path.exists(f"{prefix}_ref.npy"): continue
            
            ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
            f_ref = (convolve(ref, hs, mode='same') / (np.max(np.abs(ref)) + 1e-7)) * 0.7
            mic = (mic / (np.max(np.abs(mic)) + 1e-7)) * 0.7
            
            X_np = np.array([f_ref[i:i+512] for i in range(0, len(f_ref)-512, 128)])
            X = torch.from_numpy(X_np).unsqueeze(-1).float().to(device)
            u = model(X).cpu().detach().numpy()
            
            rec, gen = mic[512+7:], u[:, -1, 0]
            min_l = min(len(rec), len(gen))
            res = rec[:min_l] + gen[:min_l]
            
            db_gain = 10 * np.log10(np.mean(res**2) / (np.mean(rec[:min_l]**2) + 1e-12))
            print(f"Scenario {scn_id}: System Gain = {db_gain:.2f} dB")
            all_metrics.append((scn_id, db_gain))
            
            if plot_data is None: # Store the first successful scenario for the report
                plot_data = (rec[:min_l], gen[:min_l], res, scn_id)

    # --- GENERATING THE COMPREHENSIVE PNG ---
    if plot_data:
        rec, gen, res, s_id = plot_data
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # 1. Time Domain Comparison
        axs[0].plot(rec[:400], label="Original Noise", alpha=0.5, color='blue')
        axs[0].plot(gen[:400], label="v20 Gated Anti-Noise", color='red', linestyle='--')
        axs[0].plot(res[:400], label="Residual Error", color='green', linewidth=2)
        axs[0].set_title(f"Scenario {s_id}: Phase Alignment (Destructive Check)")
        axs[0].legend(); axs[0].grid(True)

        # 2. Spectral Suppression (PSD)
        f, p_orig = welch(rec, FS, nperseg=1024)
        _, p_res = welch(res, FS, nperseg=1024)
        axs[1].semilogy(f, p_orig, label="Original PSD", color='blue')
        axs[1].semilogy(f, p_res, label="Cancelled PSD", color='green')
        axs[1].fill_between(f, p_res, p_orig, where=(p_res < p_orig), color='green', alpha=0.3)
        axs[1].set_title("Spectral Power Reduction (Engine Zone 50-400Hz)")
        axs[1].set_xlim(0, 2000); axs[1].set_ylabel("Power/Freq"); axs[1].legend(); axs[1].grid(True)

        # 3. Summary Results Table
        cell_text = [[f"Scenario {m[0]}", f"{m[1]:.2f} dB"] for m in all_metrics]
        axs[2].axis('tight'); axs[2].axis('off')
        table = axs[2].table(cellText=cell_text, colLabels=["Test Case", "System Gain"], loc='center')
        table.set_fontsize(14); table.scale(1, 2)
        
        plt.tight_layout()
        plt.savefig("v20_gated_blind_test_report.png")
        print(f"\nDiagnostic report saved to 'v20_gated_blind_test_report.png'")

if __name__ == "__main__":
    run_v20_test()