# ANC Project Evolution Report: From Spectral Bands to Gated TCN (v20.1)

## 1. Executive Summary
This report documents the transition from traditional frequency-domain spectral splitting to advanced temporal sequence modeling. We moved from a **Banded Frequency approach** to a **Gated Temporal Convolutional Network (TCN)** to solve the fundamental physical challenges of Active Noise Cancellation (ANC) in a car cabin.

---

## 2. Phase 1: The "Spectral Band Division" Approach (The Failed Pilot)
Initially, we attempted to solve the ANC problem by manually dividing the acoustic spectrum into distinct frequency bands.

### The Methodology:
* **Multi-Branch Architecture:** Parallel network branches were created to handle different frequencies (e.g., a "Drone Branch" for low-frequency engine noise and a "Road Branch" for higher-frequency broadband noise).
* **Objective:** Specialize weights to handle the specific periodic nature of engine harmonics separately from stochastic road noise.

### Why it Failed:
* **Phase Incoherence at Crossover:** Dividing the spectrum created "seams" at the boundaries. Because sound is a continuous wave, the model couldn't reconcile the phase transitions between bands, leading to artifacts.
* **Summation Errors:** Branches often produced signals that were slightly out of sync, causing **Constructive Interference (+7 dB)** instead of cancellation.



---

## 3. Phase 2: Transition to Temporal Convolutional Networks (TCN)
Following the research paper *"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"*, we pivoted to a temporal-first approach using **Dilated Causal Convolutions**.

### Key ML Derivations:
* **Dilated Convolutions:** Instead of splitting frequencies, we used dilations ($1, 2, 4, 8$). This allowed the network to have a massive **Receptive Field** (seeing long-period drone waves) while maintaining a small parameter count.
* **Causal Integrity:** We ensured the model only uses "past" information to predict "anti-noise," respecting the physical speed of sound.



---

## 4. Derived Engineering Insights & v20.1 Breakthroughs

### A. The "Linear Swing" vs. Non-Linear Clipping
In v20.1, we transitioned to a "Full-Swing" `tanh` approach. However, we identified a critical trade-off:
* **The Benefit:** Unlike the ReLU (v18) which clipped the bottom half of the wave, `tanh` allows for a bi-directional signal (-1 to 1).
* **The Non-Linear Trap:** We discovered that if the TCN internal signals exceed an amplitude of **1.0**, the `tanh` enters its **Saturation Zone**. 
* **The Result:** This "squashing" of the peaks creates non-linear harmonic distortion (turning sine waves into square-ish waves). This introduces high-frequency "hissing" items that weren't in the original noise.
* **The Fix:** We implemented **Weight Normalization** and capped the `active_gain` to ensure the signals stay within the **Linear Region** of the `tanh` (roughly $-0.5$ to $0.5$).



### B. Gated Linear Units (GLU) and "Head Movement"
To solve the "Phase Drift" during live demos (where the driver moves their head, changing the physical distance), we implemented **Gating Mechanisms**.
* **The Solution:** A dual-path convolution where a **Sigmoid Gate** acts as a temporal mask. If the model detects that its prediction is shifting out of phase (due to a change in the acoustic path), the gate suppresses the output to prevent constructive amplification.



### C. Receptive Field vs. Physical Latency
* **Physical Distance:** $1.8$ meters $\approx 5.2$ ms delay $\approx 42$ samples.
* **TCN Receptive Field:** Our 4-layer stack provides **$90$ samples** of history.
* **Result:** The model "sees" the noise $11.25$ ms before it reaches the ear, allowing it to calculate the perfect anti-noise wave and its primary reflections.

---

## 5. Comparison: Banded vs. Gated TCN

| Feature | Spectral Banded (v1-v16) | Gated TCN (v20.1) |
| :--- | :--- | :--- |
| **Domain** | Frequency (Manual) | **Time (Learned)** |
| **Phase Accuracy** | Poor at crossovers | **Superior (Continuous)** |
| **Non-Linearity** | High (Branch clipping) | **Controlled (Linear Tanh)** |
| **Head Movement** | Fragile | **Robust (Gated suppression)** |
| **Final Loss** | ~0.70+ | **~0.28** |

---

## 6. Current Technical Status
The project is at **v20.1**. By enforcing a **Strict Phase Loss** (150x penalty for constructive interference) and maintaining signals in the linear `tanh` region, the system effectively suppresses engine harmonics in the $50-400$ Hz range without introducing non-linear artifacts.