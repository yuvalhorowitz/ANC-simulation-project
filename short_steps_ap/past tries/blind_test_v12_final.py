import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

# --- 1. GPU Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS, WINDOW_SIZE = 8000, 512

# --- 2. Model Architecture (Must match v12 Training exactly) ---
class SpectralExpertTCN_v12(nn.Module):
    def __init__(self):
        super(SpectralExpertTCN_v12, self).__init__()
        # Drone Branch (50-400Hz) - Massive Kernel 31
        self.drone_branch = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=31, padding=15, dilation=1),
            nn.ReLU(),
            nn.Conv1d(24, 24, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(24, 16, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1), nn.Tanh()
        )
        # Mid Branch (400-1200Hz)
        self.mid_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1), nn.Tanh()
        )
        # High Branch (1200-2000Hz) - Linear
        self.high_branch = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8), nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=1)
        )
        # Learnable Gain
        self.global_gain = nn.Parameter(torch.ones(1) * 0.8)

    def forward(self, x):
        x_in = x.transpose(1, 2)
        y_d = self.drone_branch(x_in) 
        y_m = self.mid_branch(x_in)
        y_h = self.high_branch(x_in)
        combined = (y_d * 0.7) + (y_m * 0.2) + (y_h * 0.1)
        return (combined * self.global_gain).transpose(1, 2)

# --- 3. Blind Check Execution ---
def run_v12_blind_check():
    MODEL_PATH = "anc_v12_adaptive.pth"
    DATA_DIR = "driver_bulk_4spk_data"
    TEST_SCENARIOS = range(45, 50)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Run training first.")
        return

    model = SpectralExpertTCN_v12().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    print(f"--- Running v12 Blind Check (Adaptive Gain: {model.global_gain.item():.3f}) ---")
    
    results = []
    with torch.no_grad():
        for scn_id in TEST_SCENARIOS:
            prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
            if not os.path.exists(f"{prefix}_ref.npy"): continue
            
            ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
            
            # Normalization (0.7 guard)
            f_ref = convolve(ref, hs, mode='same')
            f_ref = (f_ref / (np.max(np.abs(f_ref)) + 1e-7)) * 0.7
            mic = (mic / (np.max(np.abs(mic)) + 1e-7)) * 0.7
            
            X_list = [f_ref[i:i+WINDOW_SIZE] for i in range(0, len(f_ref)-WINDOW_SIZE, 128)]
            X_tensor = torch.tensor(np.array(X_list)).unsqueeze(-1).float().to(device)
            
            u_pred = model(X_tensor).cpu().numpy()
            
            # Reconstruct and Align
            rec = mic[WINDOW_SIZE:]
            gen = u_pred[:, -1, 0]
            min_len = min(len(rec), len(gen))
            rec, gen = rec[:min_len], gen[:min_len]
            res = rec + gen
            
            # Performance Metrics
            f, p_orig = welch(rec, FS, nperseg=256)
            _, p_resid = welch(res, FS, nperseg=256)
            
            total_db = 10 * np.log10(np.mean(rec**2) / np.mean(res**2))
            
            # Engine Zone (50-400Hz) Reduction
            mask_drone = (f >= 50) & (f <= 400)
            db_drone = 10 * np.log10(np.sum(p_orig[mask_drone]) / np.sum(p_resid[mask_drone]))
            
            # High-Frequency Zone (Whistle check)
            mask_hf = (f >= 1600) & (f <= 2000)
            db_hf = 10 * np.log10(np.sum(p_orig[mask_hf]) / np.sum(p_resid[mask_hf]))
            
            results.append({'id': scn_id, 'total': total_db, 'drone': db_drone, 'hf': db_hf})
            print(f"Scenario {scn_id}: Total={total_db:.2f}dB | Drone={db_drone:.2f}dB | HF Zone={db_hf:.2f}dB")

    # --- 4. Final Success Summary ---
    print("\n" + "="*40)
    avg_total = np.mean([r['total'] for r in results])
    print(f"Final Result: {'SUCCESS' if avg_total > 0 else 'STILL NEGATIVE'}")
    print(f"Average System Improvement: {avg_total:.2f} dB")
    print("="*40)

if __name__ == "__main__":
    run_v12_blind_check()