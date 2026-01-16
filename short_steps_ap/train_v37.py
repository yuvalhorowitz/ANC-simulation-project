import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import weight_norm
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = 8000
BATCH_SIZE = 32
EPOCHS = 150
WINDOW_SIZE = 512
LATENCY_COMP = 10 

# --- 2. Dynamic Physics ---
class DynamicPhysicsLayer(nn.Module):
    def __init__(self, hs_list, device):
        super(DynamicPhysicsLayer, self).__init__()
        self.hs_bank = []
        self.k_len = 0
        for hs_path in hs_list:
            if os.path.exists(hs_path):
                h = np.load(hs_path)
                h = h / (np.max(np.abs(h)) + 1e-7)
                self.hs_bank.append(torch.from_numpy(h).float().to(device))
                self.k_len = len(h)
        if not self.hs_bank: raise ValueError("No Hs files!")

    def forward(self, electrical_signal):
        x = electrical_signal.transpose(1, 2)
        hs_idx = np.random.randint(0, len(self.hs_bank))
        hs_kernel = self.hs_bank[hs_idx].view(1, 1, -1)
        padding = (self.k_len - 1, 0) 
        x_padded = F.pad(x, padding)
        return F.conv1d(x_padded, hs_kernel).transpose(1, 2)

# --- 3. Architecture: Cascaded MC-TCN ---
class TCNBlock(nn.Module):
    def __init__(self, in_c, out_c, k, d):
        super(TCNBlock, self).__init__()
        self.padding = (k - 1) * d
        self.conv = weight_norm(nn.Conv1d(in_c, out_c, k, padding=self.padding, dilation=d))
        self.relu = nn.PReLU() # PReLU is good for audio
        self.norm = nn.LayerNorm(out_c) # LayerNorm helps stability

    def forward(self, x):
        # x: [Batch, Time, Chan]
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out = out[:, :-self.padding, :] # Causal Crop
        out = self.relu(out)
        return self.norm(out) + x # Residual

class CascadedTCN_v37(nn.Module):
    def __init__(self):
        super(CascadedTCN_v37, self).__init__()
        
        # --- STAGE 1: MAGNITUDE CORE ---
        # Predicts a Gain Mask (How loud?)
        self.mag_input = nn.Linear(1, 32)
        self.mag_tcn = nn.Sequential(
            TCNBlock(32, 32, 5, 1),
            TCNBlock(32, 32, 5, 2),
            TCNBlock(32, 32, 5, 4),
            TCNBlock(32, 32, 5, 8)
        )
        self.mag_head = nn.Linear(32, 1)
        
        # --- STAGE 2: PHASE CORE ---
        # Predicts Phase Shift (What direction?)
        self.phase_input = nn.Linear(1, 32)
        self.phase_tcn = nn.Sequential(
            TCNBlock(32, 32, 5, 1),
            TCNBlock(32, 32, 5, 2),
            TCNBlock(32, 32, 5, 4),
            TCNBlock(32, 32, 5, 8)
        )
        self.phase_head = nn.Linear(32, 1)
        
        # Global Gain Init
        self.global_gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: [Batch, Time, 1]
        
        # 1. Magnitude Estimation
        m = self.mag_input(x)
        m = self.mag_tcn(m)
        # Softplus ensures positive gain mask
        mag_mask = F.softplus(self.mag_head(m))
        
        # Apply Gain to input (This is the "Loudness Adjusted" signal)
        x_mag = x * mag_mask
        
        # 2. Phase Estimation
        # We process the loudness-adjusted signal to find phase
        p = self.phase_input(x_mag)
        p = self.phase_tcn(p)
        # Tanh allows -1 to 1 inversion
        phase_mod = torch.tanh(self.phase_head(p))
        
        # 3. Combine
        # Hard Inversion (-1.0) * Magnitude * PhaseMod
        out = -1.0 * x_mag * phase_mod * self.global_gain
        
        return out

# --- 4. Loss Function ---
class CascadedLoss(nn.Module):
    def forward(self, electrical_pred, noise_target, physics_layer):
        # 1. Physics
        acoustic_anti_noise = physics_layer(electrical_pred)
        residual = noise_target + acoustic_anti_noise
        
        # 2. MSE (Standard)
        mse_loss = torch.mean(residual ** 2)
        
        # 3. Energy Regularization
        # We don't want the model to explode. 
        # Punish if Anti-Noise is > 2x louder than Noise
        anti_pow = torch.mean(acoustic_anti_noise**2)
        noise_pow = torch.mean(noise_target**2)
        explosion_penalty = torch.relu(anti_pow - 2.0 * noise_pow) * 10.0
        
        return mse_loss + explosion_penalty

# --- 5. Execution ---
def load_v37_data(data_dir, scenarios):
    X, y, hs_paths = [], [], []
    for scn_id in scenarios:
        p = os.path.join(data_dir, f"scn_{scn_id:03d}_hs.npy")
        if os.path.exists(p): hs_paths.append(p)
    
    for scn_id in scenarios:
        prefix = os.path.join(data_dir, f"scn_{scn_id:03d}")
        if not os.path.exists(f"{prefix}_ref.npy"): continue
        ref, mic = np.load(f"{prefix}_ref.npy"), np.load(f"{prefix}_mic.npy")
        ref = (ref / (np.max(np.abs(ref)) + 1e-7)) * 0.9
        mic = (mic / (np.max(np.abs(mic)) + 1e-7)) * 0.9
        for j in range(0, len(ref) - WINDOW_SIZE - LATENCY_COMP, 128):
            X.append(ref[j : j+WINDOW_SIZE])
            y.append(mic[j + LATENCY_COMP : j + WINDOW_SIZE + LATENCY_COMP])
            
    return torch.from_numpy(np.array(X)).unsqueeze(-1).float(), \
           torch.from_numpy(np.array(y)).unsqueeze(-1).float(), \
           list(set(hs_paths))

if __name__ == "__main__":
    X_train, y_train, hs_list = load_v37_data("driver_bulk_4spk_data_v3", range(40))
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    model = CascadedTCN_v37().to(device)
    physics = DynamicPhysicsLayer(hs_list, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = CascadedLoss()
    
    print(f"--- Training v37 (Cascaded Mag/Phase TCN) ---")
    
    history = []
    for epoch in range(EPOCHS):
        model.train()
        l_sum = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            voltage = model(data)
            loss = criterion(voltage, target, physics)
            loss.backward()
            optimizer.step()
            l_sum += loss.item()
        
        avg = l_sum/len(loader)
        history.append(avg)
        if (epoch+1)%10 == 0:
            print(f"Epoch {epoch+1} | Loss: {avg:.5f}")

    torch.save(model.state_dict(), "anc_v37_cascaded.pth")
    plt.plot(history); plt.savefig("v37_loss.png")