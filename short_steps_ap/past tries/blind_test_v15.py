import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS, WINDOW_SIZE, LATENCY_COMP = 8000, 512, 7

class PhaseSurgeonTCN_v15_1(nn.Module):
    def __init__(self):
        super(PhaseSurgeonTCN_v15_1, self).__init__()
        self.drone_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, padding=15),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 32, kernel_size=7, padding=6, dilation=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 1, kernel_size=1), nn.Tanh()
        )
        self.high_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.LeakyReLU(0.1),
            nn.Conv1d(16, 1, kernel_size=1)
        )
        self.raw_gain = nn.Parameter(torch.ones(1) * 0.0)

    def forward(self, x):
        x_in = x.transpose(1, 2)
        y_d = self.drone_branch(x_in) * 0.80
        y_h = self.high_branch(x_in) * 0.20
        active_gain = 0.4 + 0.8 * torch.sigmoid(self.raw_gain)
        return ((y_d + y_h) * active_gain).transpose(1, 2)

def run_v15_1_test():
    model = PhaseSurgeonTCN_v15_1().to(device)
    model.load_state_dict(torch.load("anc_v15_1_stabilized.pth", map_location=device))
    model.eval()

    print("--- v15.1 Surgeon Blind Test (Negative dB = SUCCESS) ---")
    results = []
    with torch.no_grad():
        for scn_id in range(45, 50):
            prefix = f"driver_bulk_4spk_data/scn_{scn_id:03d}"
            if not os.path.exists(f"{prefix}_ref.npy"): continue
            ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
            
            f_ref = (convolve(ref, hs, mode='same') / (np.max(np.abs(ref)) + 1e-7)) * 0.7
            mic = (mic / (np.max(np.abs(mic)) + 1e-7)) * 0.7
            
            X_np = np.array([f_ref[i:i+WINDOW_SIZE] for i in range(0, len(f_ref)-WINDOW_SIZE, 128)])
            X = torch.from_numpy(X_np).unsqueeze(-1).float().to(device)
            u = model(X).cpu().detach().numpy()
            
            rec, gen = mic[WINDOW_SIZE+LATENCY_COMP:], u[:, -1, 0]
            min_l = min(len(rec), len(gen))
            res = rec[:min_l] + gen[:min_l]
            
            db_gain = 10 * np.log10(np.mean(res**2) / (np.mean(rec[:min_l]**2) + 1e-12))
            print(f"Scenario {scn_id}: System Gain = {db_gain:.2f} dB")
            
            f, p_orig = welch(rec[:min_l], FS, nperseg=256)
            _, p_resid = welch(res, FS, nperseg=256)
            results.append({'f': f, 'p_orig': p_orig, 'p_resid': p_resid, 'total': db_gain, 'id': scn_id})

    # Visualization of the best Scenario
    target = min(results, key=lambda x: x['total'])
    plt.figure(figsize=(12, 6))
    plt.semilogy(target['f'], target['p_orig'], label="Original Noise")
    plt.semilogy(target['f'], target['p_resid'], color='green', label="v15.1 Residual")
    plt.fill_between(target['f'], target['p_resid'], target['p_orig'], where=(target['p_resid'] < target['p_orig']), color='green', alpha=0.2)
    plt.title(f"v15.1 Final Result: Scenario {target['id']} ({target['total']:.2f} dB)")
    plt.xlim(0, 2200); plt.grid(True); plt.legend()
    plt.savefig("v15_1_blind_test_report.png")

if __name__ == "__main__":
    run_v15_1_test()