# 📝 Lessons Learned: Active Noise Control (ANC) with TCN

## 1. Sampling Rate & Signal Requirements
* [cite_start]**From 16kHz to 8kHz:** Lowering the sampling rate to **8kHz** doubled our computational efficiency[cite: 471]. 
* **Cycle Calculation (50Hz):** To cancel a 50Hz engine drone at 8kHz, one full cycle requires:  
  $$\frac{1}{50} \times 8000 = 160 \text{ samples}$$
* [cite_start]**Receptive Field (RF):** To identify the noise pattern, the TCN must "see" at least one full cycle (160 samples) within its history[cite: 332, 493].

---

## 2. TCN Architectural Theory (Based on Bai et al., 2018)
* **Receptive Field Formula:** The effective history size of a TCN is defined by:  
  $$RF = 1 + (k-1) \cdot (2^L - 1)$$  
  [cite_start]where $k$ is the kernel size and $L$ is the number of layers[cite: 146].
* [cite_start]**Optimized Parameters ($k=5, L=6$):** * Using a **Kernel Size ($k$) of 5** and **6 layers ($L$)** results in an RF of **253 samples**[cite: 146].
  * This RF (approx. 31.6ms) safely covers the 160-sample requirement for the 50Hz fundamental frequency.
* [cite_start]**Causal Constraint:** TCNs use **Causal Convolutions**, ensuring that an output at time $t$ depends only on inputs from $t$ and earlier, preventing future data leakage[cite: 73, 86, 91].
* [cite_start]**Dilated Convolutions:** Exponentially increasing dilations ($d = 2^i$) allow the network to achieve a large receptive field without requiring an extremely deep architecture[cite: 75, 103].



---

## 3. Practical Discovery: Simple vs. Complex Architectures
* [cite_start]**Over-Engineering Pitfall:** While the TCN paper suggests **Residual Blocks** and **Batch Normalization** for complex tasks like Language Modeling[cite: 75, 150, 157], these proved counterproductive for simple periodic noise cancellation.
* [cite_start]**Phase Distortion:** Techniques like **Batch Normalization** and **Spatial Dropout** can distort the amplitude and phase information of the signal[cite: 160, 161, 163].
* **Stability Conclusion:** For deterministic signals (engine harmonics), a **Simple Causal Stack** of 1D convolutions converges faster and avoids the "Loss Stagnation" (stuck at 2.4/2.5) observed in complex models.

---

## 4. Training Stability & Optimization
* [cite_start]**Gradient Clipping:** The paper recommends **Gradient Clipping** (threshold [0.3, 1]) to prevent gradient explosion in sequence modeling[cite: 232, 496].
* [cite_start]**Activation Functions:** * **ReLU:** Used in hidden layers for efficient gradient propagation[cite: 159, 527].
  * **Tanh:** Critical for the output layer to ensure the generated anti-noise signal respects physical speaker constraints within the range $[-1, 1]$.
* **Pre-Filtering (System ID):** Convolving the reference signal with the **Room Impulse Response (RIR)** before training allows the model to learn the inverse filter of the acoustic environment directly.



---

## 5. Summary of Optimal ANC Configuration
| Feature | Setting | Rationale |
| :--- | :--- | :--- |
| **Sampling Rate ($F_s$)** | 8000 Hz | High enough for engine noise; low enough for memory efficiency. |
| **Kernel Size ($k$)** | 5 | [cite_start]Provides better phase stability than $k=3$[cite: 513]. |
| **Num Layers ($L$)** | 6 | Ensures the RF covers the low-frequency drone (50Hz). |
| **Architecture** | Simple TCN | Prevents phase distortion caused by BN/Residuals in periodic tasks. |
| **Loss Function** | MSE | [cite_start]Direct optimization for time-domain signal matching[cite: 87, 191]. |