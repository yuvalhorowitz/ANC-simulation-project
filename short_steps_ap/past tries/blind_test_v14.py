import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS, WINDOW_SIZE, LATENCY_COMP = 8000, 512, 7

class PhaseSurgeonTCN_v14(nn.Module):
    # (Same Architecture as Trainer)
    def __init__(self):
        super(PhaseSurgeonTCN_v14, self).__init__()
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
        self.global_gain = nn.Parameter(torch.ones(1) * 0.9)

    def forward(self, x):
        x_in = x.transpose(1, 2)
        y_d = self.drone_branch(x_in) * 0.80
        y_h = self.high_branch(x_in) * 0.20
        return ((y_d + y_h) * self.global_gain).transpose(1, 2)

def run_v14_test():
    model = PhaseSurgeonTCN_v14().to(device)
    model.load_state_dict(torch.load("anc_v14_surgeon.pth", map_location=device, weights_only=True))
    model.eval()

    print("--- v14 Surgeon Blind Test (Negative dB = SUCCESS) ---")
    for scn_id in range(45, 50):
        prefix = f"driver_bulk_4spk_data/scn_{scn_id:03d}"
        if not os.path.exists(f"{prefix}_ref.npy"): continue
        ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
        f_ref = (convolve(ref, hs, mode='same') / (np.max(np.abs(ref)) + 1e-7)) * 0.7
        mic = (mic / (np.max(np.abs(mic)) + 1e-7)) * 0.7
        
        X = torch.tensor([f_ref[i:i+512] for i in range(0, len(f_ref)-512, 128)]).unsqueeze(-1).float().to(device)
        u = model(X).cpu().detach().numpy()
        rec, gen = mic[512+LATENCY_COMP:], u[:, -1, 0]
        min_l = min(len(rec), len(gen))
        res = rec[:min_l] + gen[:min_l]
        
        # SUCCESS METRIC: Negative dB means the residual is smaller than original
        db_gain = 10 * np.log10(np.mean(res**2) / (np.mean(rec[:min_l]**2) + 1e-12))
        print(f"Scenario {scn_id}: System Gain = {db_gain:.2f} dB")

if __name__ == "__main__":
    run_v14_test()