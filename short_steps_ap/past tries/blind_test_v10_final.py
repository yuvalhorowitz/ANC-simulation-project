import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

# --- 1. GPU Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS, WINDOW_SIZE = 8000, 512

# --- 2. Model Architecture (Must match v10 Training exactly) ---
class SpectralExpertTCN(nn.Module):
    def __init__(self):
        super(SpectralExpertTCN, self).__init__()
        self.drone_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1), nn.Tanh()
        )
        self.mid_branch = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=7, padding=3),
            nn.BatchNorm1d(12), nn.ReLU(),
            nn.Conv1d(12, 1, kernel_size=1), nn.Tanh()
        )
        self.high_branch = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8), nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=1)
        )

    def forward(self, x):
        x_in = x.transpose(1, 2)
        y_d = self.drone_branch(x_in) * 0.4
        y_m = self.mid_branch(x_in) * 0.3
        y_h = self.high_branch(x_in) * 0.3
        return (y_d + y_m + y_h).transpose(1, 2)

# --- 3. Blind Test Logic ---
def run_v10_blind_test():
    MODEL_PATH = "anc_v10_spectral_experts.pth"
    DATA_DIR = "driver_bulk_4spk_data"
    TEST_SCENARIOS = range(45, 50)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Finish training first!")
        return

    model = SpectralExpertTCN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    print(f"--- Running v10 Blind Test on: {device} ---")
    
    results = []

    with torch.no_grad():
        for scn_id in TEST_SCENARIOS:
            prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
            if not os.path.exists(f"{prefix}_ref.npy"): continue
            
            ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
            
            # --- Pre-processing (Normalization Guard) ---
            f_ref = convolve(ref, hs, mode='same')
            f_ref = (f_ref / (np.max(np.abs(f_ref)) + 1e-7)) * 0.7
            mic = (mic / (np.max(np.abs(mic)) + 1e-7)) * 0.7
            
            # Prepare sliding windows
            X_list = [f_ref[i:i+WINDOW_SIZE] for i in range(0, len(f_ref)-WINDOW_SIZE, 128)]
            X_tensor = torch.tensor(np.array(X_list)).unsqueeze(-1).float().to(device)
            
            # Inference
            u_pred = model(X_tensor).cpu().numpy()
            
            # Signal Reconstruction
            rec = mic[WINDOW_SIZE:]
            gen = u_pred[:, -1, 0] # Take the last sample of each window
            min_len = min(len(rec), len(gen))
            rec, gen = rec[:min_len], gen[:min_len]
            res = rec + gen
            
            # Spectral Performance Calculation
            f, p_orig = welch(rec, FS, nperseg=512)
            _, p_resid = welch(res, FS, nperseg=512)
            
            # 1.6k-2k Zone check (Whistle Protection)
            mask_hf = (f >= 1600) & (f <= 2000)
            db_hf = 10 * np.log10(np.sum(p_orig[mask_hf]) / np.sum(p_resid[mask_hf]))
            
            # Total Reduction
            total_db = 10 * np.log10(np.mean(rec**2) / np.mean(res**2))
            
            results.append({'id': scn_id, 'total': total_db, 'hf': db_hf, 'f': f, 'p1': p_orig, 'p2': p_resid})
            print(f"Scenario {scn_id}: Total Red.={total_db:.2f}dB | HF Zone={db_hf:.2f}dB")

    # --- 4. Comparative Visualization ---
    target = results[0] # Visualize Scenario 45
    plt.figure(figsize=(14, 7))
    plt.semilogy(target['f'], target['p1'], label="Original Noise", alpha=0.4, color='blue')
    plt.semilogy(target['f'], target['p2'], label="v10 Residual", color='green', lw=2)
    plt.fill_between(target['f'], target['p2'], target['p1'], where=(target['p2'] < target['p1']), color='green', alpha=0.2)
    plt.axvspan(1600, 2000, color='red', alpha=0.1, label='Surgical Zone (Muted)')
    
    plt.title(f"v10 Spectral Expert: Blind Scenario {target['id']}\nTotal Reduction: {target['total']:.2f} dB")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Power (dB)"); plt.xlim(0, 2200); plt.legend(); plt.grid(True)
    
    plt.savefig("v10_blind_test_report.png")
    print("\nBlind Test complete. Check 'v10_blind_test_report.png' for the spectral map.")

if __name__ == "__main__":
    run_v10_blind_test()