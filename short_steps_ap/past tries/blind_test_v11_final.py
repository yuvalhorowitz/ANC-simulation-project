import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

# --- 1. GPU Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS, WINDOW_SIZE = 8000, 512

# --- 2. Model Architecture (Must match v11 Training exactly) ---
class SpectralExpertTCN_v11(nn.Module):
    def __init__(self):
        super(SpectralExpertTCN_v11, self).__init__()
        
        # Expert 1: 50-400Hz (Drone) - Dilated TCN for Phase Memory
        self.drone_branch = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1, dilation=1), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=8, dilation=8), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=16, dilation=16), 
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1), nn.Tanh()
        )
        
        self.mid_branch = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=3, padding=1, dilation=1), nn.ReLU(),
            nn.Conv1d(12, 12, kernel_size=3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(12, 12, kernel_size=3, padding=4, dilation=4), 
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
        # Match v11 Power Weights (60/25/15)
        y_d = self.drone_branch(x_in) * 0.6
        y_m = self.mid_branch(x_in) * 0.25
        y_h = self.high_branch(x_in) * 0.15
        return (y_d + y_m + y_h).transpose(1, 2)

# --- 3. Evaluation Suite ---
def run_v11_blind_check():
    MODEL_PATH = "anc_v11_phase_locked.pth"
    DATA_DIR = "driver_bulk_4spk_data"
    TEST_SCENARIOS = range(45, 50)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Ensure training is finished.")
        return

    model = SpectralExpertTCN_v11().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    print(f"--- Running v11 Blind Check (Dilated Memory) ---")
    
    results = []

    with torch.no_grad():
        for scn_id in TEST_SCENARIOS:
            prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
            if not os.path.exists(f"{prefix}_ref.npy"): continue
            
            ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
            
            # Normalization Guard (Consistent with Trainer)
            f_ref = convolve(ref, hs, mode='same')
            f_ref = (f_ref / (np.max(np.abs(f_ref)) + 1e-7)) * 0.7
            mic = (mic / (np.max(np.abs(mic)) + 1e-7)) * 0.7
            
            X_list = [f_ref[i:i+WINDOW_SIZE] for i in range(0, len(f_ref)-WINDOW_SIZE, 128)]
            X_tensor = torch.tensor(np.array(X_list)).unsqueeze(-1).float().to(device)
            
            # Inference
            u_pred = model(X_tensor).cpu().numpy()
            
            # Reconstruct
            rec = mic[WINDOW_SIZE:]
            gen = u_pred[:, -1, 0]
            min_len = min(len(rec), len(gen))
            rec, gen = rec[:min_len], gen[:min_len]
            res = rec + gen
            
            # Metrics
            f, p_orig = welch(rec, FS, nperseg=256)
            _, p_resid = welch(res, FS, nperseg=256)
            
            total_db = 10 * np.log10(np.mean(rec**2) / np.mean(res**2))
            
            # 1.6k-2k Zone Suppression
            mask_hf = (f >= 1600) & (f <= 2000)
            db_hf = 10 * np.log10(np.sum(p_orig[mask_hf]) / np.sum(p_resid[mask_hf]))
            
            results.append({'id': scn_id, 'total': total_db, 'hf': db_hf, 'f': f, 'p_orig': p_orig, 'p_resid': p_resid})
            print(f"Scn {scn_id}: Total Red.={total_db:.2f}dB | HF Zone={db_hf:.2f}dB")

    # --- 4. Plotting Scenario 45 Performance ---
    target = results[0]
    plt.figure(figsize=(12, 6))
    plt.semilogy(target['f'], target['p_orig'], label="Original Noise", alpha=0.5, color='blue')
    plt.semilogy(target['f'], target['p_resid'], label="v11 Residual", color='green', lw=2)
    plt.axvspan(1600, 2000, color='red', alpha=0.1, label='Surgical Zone')
    
    plt.title(f"v11 Phase-Locked: Blind Test (Scenario {target['id']})\nTotal Red: {target['total']:.2f} dB")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Power (dB)"); plt.xlim(0, 2200); plt.legend(); plt.grid(True)
    
    plt.savefig("v11_blind_test_final_report.png")
    print("\nVisual report saved. If Total Red is positive, the Phase-Locking is working.")

if __name__ == "__main__":
    run_v11_blind_check()