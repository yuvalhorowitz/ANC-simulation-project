import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve, welch

# --- 1. GPU Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS, WINDOW_SIZE = 8000, 512
LATENCY_COMP = 7 # Must match training script

# --- 2. Model Architecture (Exact match to v13) ---
class PredictiveExpertTCN_v13(nn.Module):
    def __init__(self):
        super(PredictiveExpertTCN_v13, self).__init__()
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
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=1)
        )
        self.global_gain = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, x):
        x_in = x.transpose(1, 2)
        y_d = self.drone_branch(x_in) * 0.85
        y_h = self.high_branch(x_in) * 0.15
        return ((y_d + y_h) * self.global_gain).transpose(1, 2)

# --- 3. Blind Test Logic ---
def run_v13_blind_test():
    MODEL_PATH = "anc_v13_predictive.pth"
    DATA_DIR = "driver_bulk_4spk_data"
    TEST_SCENARIOS = range(45, 50)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Ensure v13 training is complete.")
        return

    model = PredictiveExpertTCN_v13().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    print(f"--- Running v13 Predictive Blind Check (Gain: {model.global_gain.item():.2f}) ---")
    
    scenario_data = []

    with torch.no_grad():
        for scn_id in TEST_SCENARIOS:
            prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
            if not os.path.exists(f"{prefix}_ref.npy"): continue
            
            ref, mic, hs = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy"), np.load(f"{prefix}_hs.npy")
            
            # Normalization (0.7 guard)
            f_ref = convolve(ref, hs, mode='same')
            f_ref = (f_ref / (np.max(np.abs(f_ref)) + 1e-7)) * 0.7
            mic = (mic / (np.max(np.abs(mic)) + 1e-7)) * 0.7
            
            # Note: In inference, we don't 'shift' the data, the model 
            # naturally outputs the leading phase it learned during training.
            X_list = [f_ref[i:i+WINDOW_SIZE] for i in range(0, len(f_ref)-WINDOW_SIZE, 128)]
            X_tensor = torch.tensor(np.array(X_list)).unsqueeze(-1).float().to(device)
            
            u_pred = model(X_tensor).cpu().numpy()
            
            # Reconstruction (Alignment Fix)
            # Because of LATENCY_COMP during training, the model's output at index 'j' 
            # corresponds to the mic signal at 'j + LATENCY_COMP'.
            rec = mic[WINDOW_SIZE + LATENCY_COMP:]
            gen = u_pred[:, -1, 0]
            min_len = min(len(rec), len(gen))
            rec, gen = rec[:min_len], gen[:min_len]
            res = rec + gen
            
            # Spectral Metrics
            f, p_orig = welch(rec, FS, nperseg=256)
            _, p_resid = welch(res, FS, nperseg=256)
            
            total_db = 10 * np.log10(np.mean(rec**2) / np.mean(res**2))
            mask_drone = (f >= 50) & (f <= 400)
            db_drone = 10 * np.log10(np.sum(p_orig[mask_drone]) / np.sum(p_resid[mask_drone]))
            
            scenario_data.append({'id': scn_id, 'total': total_db, 'drone': db_drone, 
                                  'f': f, 'p_orig': p_orig, 'p_resid': p_resid,
                                  'rec_sig': rec, 'res_sig': res})
            
            print(f"Scenario {scn_id}: Total Red.={total_db:.2f}dB | Engine Zone={db_drone:.2f}dB")

    # --- 4. Plotting Detailed Report ---
    target = scenario_data[0] # Visualize Scenario 45
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Spectral Plot
    ax1.semilogy(target['f'], target['p_orig'], label="Original Noise", alpha=0.5, color='blue')
    ax1.semilogy(target['f'], target['p_resid'], label="v13 Predictive Residual", color='green', lw=2)
    ax1.fill_between(target['f'], target['p_resid'], target['p_orig'], where=(target['p_resid'] < target['p_orig']), color='green', alpha=0.2)
    ax1.set_title(f"v13 Blind Test Scenario {target['id']}: Spectral Suppression\nTotal Reduction: {target['total']:.2f} dB")
    ax1.set_xlim(0, 2200); ax1.legend(); ax1.grid(True)

    # Time Domain Zoom Plot
    ax2.plot(target['rec_sig'][1000:1400], label="Mic Signal (Noise)", color='blue', alpha=0.4)
    ax2.plot(target['res_sig'][1000:1400], label="Residual (Error)", color='red', lw=1.5)
    ax2.set_title("Time Domain Analysis: Phase Cancellation Success")
    ax2.set_xlabel("Samples"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig("v13_blind_test_report.png")
    print("\nVisual report saved. Analyze 'v13_blind_test_report.png' to verify phase alignment.")

if __name__ == "__main__":
    run_v13_blind_test()