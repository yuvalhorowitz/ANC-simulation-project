# ANC Project Development Report: The Evolution to v20.1

## 1. Project Objective
The goal of this project is to develop a Deep Learning-based **Active Noise Cancellation (ANC)** system designed to suppress engine drone ($50-400$ Hz) and road noise within a simulated car cabin. The system is engineered to handle physical propagation delays, acoustic reflections, and the real-time constraints required for hardware deployment (e.g., TI CC3x or ARM Cortex).

---

## 2. DSP Foundations: The Physics of the Problem
Before applying Machine Learning, we established the digital signal processing (DSP) requirements based on the acoustic environment.

### A. Phase Inversion & Superposition
The core principle is destructive interference. We generate an anti-noise signal $u(n)$ such that:
$$Error(n) = Noise(n) + u(n) \approx 0$$
To achieve this, $u(n)$ must be a perfect $180^\circ$ phase-shifted version of the noise.



### B. Secondary Path ($H_s$)
The anti-noise signal is played through car speakers. The acoustic path from the speaker to the driver’s ear is known as the **Secondary Path**. The model must learn to "pre-filter" the anti-noise to compensate for the frequency response and reverberation of the cabin.

### C. Latency and Receptive Field
At a sampling rate of **$8000$ Hz**, each sample represents $0.125$ ms. A physical distance of $1.8$ m between the source and the ear creates a $\sim 5.2$ ms delay (approx. $42$ samples). Our neural network's **Receptive Field** must be significantly larger than this delay to "see" the history needed for prediction.

---

## 3. The Machine Learning Evolution

### Phase 1: v1–v16 (Recurrent & Baseline Models)
Early versions utilized LSTMs and standard CNNs. These models struggled with the "Vanishing Gradient" problem and failed to capture the long-term temporal dependencies of periodic engine harmonics.

### Phase 2: v17–v18 (The TCN Transition)
Following the research paper *"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"*, we moved to a **Temporal Convolutional Network (TCN)**.

* **Discovery:** TCNs outperformed LSTMs due to **Dilated Causal Convolutions**.
* **The v18 Failure:** We followed the paper’s generic residual block which included a final **ReLU** activation.
* **Result:** The ReLU clipped the negative part of the audio wave. Since sound is a bi-directional pressure wave, this prevented phase inversion, leading to **Constructive Interference (+7 dB)**.



### Phase 3: v19 (The Linear Guard)
We removed the final ReLU to allow for a full linear range ($-1$ to $1$).
* **New Feature:** Introduced a learnable **Gain Parameter** (`active_gain`) to control the output amplitude authority.
* **Result:** The model became stable, but was still sensitive to "Phase Drift" caused by head movement in the simulation.

---

## 4. The Current Stage: v20.1 Gated TCN
Version 20.1 is our most robust architecture, incorporating advanced gating mechanisms to ensure real-world stability.

### A. Gated Linear Units (GLU)
Derived from the TCN paper's section on gating mechanisms, we implemented a dual-convolution path:
1.  **Data Path:** Extracts wave features.
2.  **Gate Path (Sigmoid):** Acts as a temporal mask.
If the model detects that its prediction is shifting out of phase (due to a change in the Secondary Path), the gate suppresses the output, preventing noise amplification.



### B. Strict Causal Slicing
To prevent "information leakage" (cheating) during training, we enforced strict causality. The output at time $t$ is calculated only from samples $0$ to $t$ by slicing the padding. This ensures the model learns the **Physical Delay** rather than just a mathematical correlation.

### C. Strict Phase Loss Function
We implemented a custom loss function that includes a **Cross-Correlation Penalty**:
* If the correlation between Anti-Noise and Noise is positive (Constructive), the loss is penalized by a **150x multiplier**.
* This forces the TCN to stay in the **Destructive Zone** ($180^\circ$ shift).

---

## 5. Summary of System Parameters

| Metric | Specification | Engineering Reason |
| :--- | :--- | :--- |
| **Sampling Rate** | $8000$ Hz | Optimization for engine drone frequencies. |
| **Activation** | `tanh` (Final Head) | Maps signal to physical -1 to 1 audio range. |
| **Dilation Factors** | $1, 2, 4, 8$ | Creates exponential receptive field ($90$ samples). |
| **Internal Gating** | `Sigmoid` (GLU) | Prevents runaway constructive interference. |
| **Loss Weight** | $15\times$ MSE | High-pressure energy matching after Epoch 60. |

---

## 6. Project Status
* **Training Loss:** Stable at **~0.28**.
* **Gain Authority:** Settled at **1.1x**, indicating accurate amplitude matching.
* **Status:** Successfully suppresses tonal engine harmonics in the **50–400 Hz** range.

---