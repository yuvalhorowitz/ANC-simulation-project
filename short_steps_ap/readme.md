# 🚀 Intermediate Summary: ANC-simulation-project
**Date:** December 27, 2025
**Current Status:** Phase 2 (Acoustic Reflection Handling)

---

## 🎯 Project Objective
Development of a Deep Learning-based **Active Noise Control (ANC)** system using **Temporal Convolutional Networks (TCN)**. The goal is to cancel engine noise (50Hz + harmonics) within a cabin environment while handling complex acoustic reflections (Reverberation).

---

## 📂 Script & Development Evolution

| File Name | Primary Focus | Key Outcome / Learning |
| :--- | :--- | :--- |
| `fast_train.py` | **Baseline TCN** | Resolved `float32/64` precision mismatches. Validated basic learning logic. |
| `fast_train_conv.py` | **Secondary Path ($H_s$)** | Integrated physical speaker-to-ear convolution into the loss function. |
| `generate_fast_data_reverb.py` | **Reverb Simulation** | Used `pyroomacoustics` ($max\_order=3$) to generate data with realistic echoes. |
| `fast_train_reverb_v2.py` | **Large Receptive Field** | Increased dilations up to $d=32$ to capture long impulse response tails. |
| `fast_train_rir_prefilter.py` | **System Identification** | **Current Champion:** Pre-filtering input with RIR allowed a compact model to achieve **~8.85 dB** reduction. |

---

## 🧠 Core Technical Implementation

### 1. Filtered-X TCN Logic
Instead of predicting noise directly, the model learns the **Inverse Filter** of the room. By pre-conditioning the input (Reference) with the room's **Impulse Response (RIR)**, we decouple the physics from the prediction logic.

### 2. Architectural Decisions
* **Causal Padding:** Essential for real-time compliance; ensures no future-data leakage.
* **Dilated Convolutions:** Used to expand the **Receptive Field** to see low-frequency cycles without increasing parameters.
* **Tanh Activation:** A physical constraint layer that prevents the model from exceeding the speaker's dynamic range.

### 3. Mathematical Evaluation
* **MSE:** Mean Squared Error in the time domain.
* **PSD (Power Spectral Density):** Evaluation in the frequency domain using $dB$ scale.
* **Reduction Formula:** $$NR (dB) = 10 \cdot \log_{10} \left( \frac{\text{Mean Square (Original)}}{\text{Mean Square (Residual)}} \right)$$

---

## ⚠️ Identified Issues: The "Waterbed Effect"
In the latest results (`anc_performance_db.png`), we observed successful cancellation at **50Hz**, but observed **amplification (Constructive Interference)** at frequencies above **250Hz**. 

* **Cause:** Phase sensitivity increases at higher frequencies. Small timing errors in the TCN result in "adding" noise instead of "subtracting" it.
* **Observation:** The model prioritizes the high-energy 50Hz peak to lower the global MSE, neglecting the lower-energy high-frequency harmonics.

---

## 🛠 Development Roadmap (Next Session)
1.  **Frequency-Weighted Loss:** Modify the loss function to heavily penalize errors in the 200Hz-500Hz range.
2.  **Kernel Size Optimization:** Increase `kernel_size` from 3 to 5 or 7 for smoother control signals.
3.  **Robustness Testing:** Introduce randomized microphone/speaker positions to move beyond static scenarios.

---
**Status:** *Paused - Ready for resumption from `fast_train_rir_prefilter.py`.*