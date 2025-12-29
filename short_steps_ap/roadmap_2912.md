# 🚀 Project Summary & Roadmap: ANC via TCN

**Date:** December 29, 2025
**Current Status:** Phase 2 (Acoustic Reflection Handling & Optimization)

---

## 🛠️ Current Technical Baseline
Our current stable model is optimized for engine noise cancellation in a reverberant environment.

* **Sampling Rate ($F_s$):** 8000 Hz. Chosen to balance Nyquist requirements with computational efficiency.
* **Architecture:** Simple TCN Stack. We moved away from Residual Blocks and Batch Normalization to prevent phase distortion and "Loss Stagnation" in periodic signal tasks.
* **Kernel Size ($k$):** 5. Provides better phase stability than $k=3$ at 8kHz.
* **Number of Layers ($L$):** 6. 
* **Receptive Field (RF):** 253 samples ($\approx 31.6ms$). Calculated using $RF = 1 + (k-1) \cdot (2^L - 1)$, ensuring it covers at least one full cycle of 50Hz (160 samples).
* **Input Conditioning:** Pre-filtering the Reference signal with the Room Impulse Response (RIR).

---

## 📈 Results & Identified Issues
* **Performance:** Achieved approximately **~8.8 dB** total noise reduction.
* **The "Waterbed Effect":** Successful cancellation at 50Hz but observed **amplification (Constructive Interference)** at frequencies between 200Hz and 500Hz.
* **Root Cause:** Increased phase sensitivity at higher frequencies. Small timing errors in the TCN output result in adding noise instead of subtracting it.

---

## 📝 Training Strategy & Lessons Learned

### Architecture Insights
* **Simple over Complex:** For deterministic signals like engine harmonics, a simple causal stack of 1D convolutions outperforms complex models with Residuals/BN, which often get stuck at a high loss (e.g., 2.44).
* **Activation Functions:** ReLU for hidden layers and **Tanh** for the output layer to respect physical speaker constraints [-1, 1].

### Optimization Techniques
* **Gradient Clipping:** Clipping between [0.3, 1] is essential to prevent gradient explosion in sequence modeling.
* **Loss Function:** Currently using **MSE** in the time domain.

---

## 🗺️ Development Roadmap

### Stage 1: Frequency Precision
* **Frequency-Weighted Loss:** Modify the loss function to heavily penalize errors in the 200Hz-500Hz range.
* **Spectral Convergence:** Integrate frequency-domain analysis into the training loop.

### Stage 2: Signal Complexity
* **Non-Stationary Noise:** Transition to dynamic RPM simulations (e.g., accelerating from 2000 to 4000 RPM).
* **Broadband Integration:** Introduce road and wind noise to test non-periodic cancellation.

### Stage 3: Robustness & Generalization
* **Multi-RIR Training:** Generate datasets with 100+ different cabin environments and microphone positions.
* **Acoustic Augmentation:** Randomize absorption coefficients to simulate different cabin materials.

### Stage 4: Engineering & Deployment
* **Latency Analysis:** Measure `model.predict` execution time to ensure real-time feasibility.
* **Memory Footprint:** Optimize RAM/GPU usage for hardware deployment.
* **Quantization:** Prepare the model for 16-bit/8-bit integer inference on DSP/FPGA platforms.

---
**Current Priority:** Resolving the high-frequency amplification issue via loss function refinement.