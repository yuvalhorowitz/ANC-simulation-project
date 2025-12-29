# 📓 Project Documentation: ANC System Development via TCN
**Author:** Ariel Turnovsky  
**Project:** BSc Final Project - Active Noise Control Simulation  
**Status:** Phase 3 - Robustness & Multi-Scenario Testing

---

## 1. Project Objective
The goal is to develop a Deep Learning-based **Active Noise Control (ANC)** system using **Temporal Convolutional Networks (TCN)** to cancel engine noise (50Hz + harmonics) in a reverberant cabin environment.

---

## 2. Phase 1: Foundation & Baseline (16kHz)
In this phase, we established the basic TCN structure and resolved fundamental technical hurdles.

* **Key Files:** `fast_train.py`, `generate_fast_data_reverb.py`.
* **Key Lessons:**
    * Resolved precision mismatches between `float32` and `float64`.
    * Implemented **PyRoomAcoustics** to simulate realistic cabin reflections (Reverberation).
    * Observed that a large **Receptive Field** is necessary to capture long impulse response tails.

---

## 3. Phase 2: Optimization & Physical Realism (8kHz)
We shifted toward a more efficient and physically accurate model.

* **Sampling Rate Transition:** Switched from 16kHz to **8kHz** to double computational efficiency and increase the effective history "memory" for the same number of parameters.
* **System Identification:** Implemented **RIR Pre-filtering**, convolving the reference signal with the Room Impulse Response (Hs) before training. This allowed the model to focus on learning the inverse filter of the room.
* **Architecture Refinement:** * Settled on a **Simple TCN Stack** (6 layers, Kernel size $k=5$).
    * Found that complex structures (Batch Normalization/Residuals) distorted phase information in simple periodic signals, leading to "Loss Stagnation" around 2.44.
* **Normalization Success:** The file `train_tcn_8k_v3_normalized.py` became the POC champion by ensuring all signals were normalized to $[-1, 1]$, preventing gradient explosion and stabilizing training.



---

## 4. Phase 3: Robustness & Generalization (Current)
Moving from a single-scenario POC to a system that works in unseen environments.

* **Identified Issue - The "Waterbed Effect":** Observed successful cancellation at 50Hz but noticed **amplification** (Constructive Interference) between 200Hz and 500Hz due to high phase sensitivity at 8kHz.
* **Multi-Scenario Training:** Developed `train_test_blind_scenario.py` to train the model on Scenarios 0 & 1 and perform a **Blind Test** on Scenario 2.
* **Metrics:** Achieved an initial benchmark of **~8.8 dB** reduction on seen data.



---

## 5. Summary of Engineering Takeaways

| Metric | Optimal Value | Engineering Rationale |
| :--- | :--- | :--- |
| **Sampling Rate** | 8000 Hz | Efficiently covers engine noise while reducing parameters. |
| **Kernel Size ($k$)** | 5 | Provides essential phase stability for 8kHz resolution. |
| **Receptive Field** | 253 samples | Covers >1.5 cycles of 50Hz noise ($RF = 1 + (k-1)(2^L-1)$). |
| **Loss Function** | MSE | Standard regression target for time-domain signal matching. |
| **Optimization** | Clipnorm=1.0 | Prevents gradient explosion in TCN sequence modeling. |

---

## 6. Future Roadmap
1. **Frequency-Weighted Loss:** Penalizing high-frequency errors to fix the amplification issue.
2. **Dynamic Noise:** Testing the model against varying RPMs (Acceleration).
3. **Hardware Readiness:** Quantizing the model for potential deployment on ARM/FPGA platforms.