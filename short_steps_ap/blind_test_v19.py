import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- v19 Architecture Identity Match ---
class CausalLinearBlock_v19(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super(CausalLinearBlock_v19, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation))
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation))
        self.relu2 = nn.LeakyReLU(0.1)
        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    def forward(self, x):
        out = self.net(x)[:, :, :-self.padding*2]
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class FinalTCN_v19(nn.Module):
    def __init__(self):
        super(FinalTCN_v19, self).__init__()
        self.tcn = nn.Sequential(
            CausalLinearBlock_v19(1, 32, kernel_size=7, dilation=1),
            CausalLinearBlock_v19(32, 32, kernel_size=7, dilation=2),
            CausalLinearBlock_v19(32, 32, kernel_size=7, dilation=4),
            CausalLinearBlock_v19(32, 32, kernel_size=7, dilation=8)
        )
        self.linear_head = nn.Conv1d(32, 1, 1)
        self.raw_gain = nn.Parameter(torch.ones(1) * 0.0)
    def forward(self, x):
        x_in = x.transpose(1, 2)
        y = self.tcn(x_in)
        y = self.linear_head(y)
        active_gain = 0.5 + 1.0 * torch.sigmoid(self.raw_gain)
        anti_noise = y * active_gain
        return (-1.0 * anti_noise).transpose(1, 2)

def run_v19_test():
    model = FinalTCN_v19().to(device)
    model.load_state_dict(torch.load("anc_v19_linear.pth", map_location=device))
    model.eval()

    print("--- v19 Linear Guard Blind Test (Negative dB = SUCCESS) ---")
    results = []
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
            
            f, p_orig = welch(rec[:min_l], 8000, nperseg=256)
            _, p_resid = welch(res, 8000, nperseg=256)
            results.append({'f': f, 'p_orig': p_orig, 'p_resid': p_resid, 'total': db_gain, 'id': scn_id})

    plt.figure(figsize=(12, 6))
    best = min(results, key=lambda x: x['total'])
    plt.semilogy(best['f'], best['p_orig'], label="Original Noise")
    plt.semilogy(best['f'], best['p_resid'], color='green', label="v19 Residual")
    plt.fill_between(best['f'], best['p_resid'], best['p_orig'], where=(best['p_resid'] < best['p_orig']), color='green', alpha=0.2)
    plt.title(f"v19 Final Result: Scenario {best['id']} ({best['total']:.2f} dB)")
    plt.xlim(0, 2200); plt.grid(True); plt.legend()
    plt.savefig("v19_linear_blind_test.png")

if __name__ == "__main__":
    run_v19_test()