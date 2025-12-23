# Experiment Log - Phase 2 New

## Baseline Reference (phase_2)

**Date**: 2024-12-14 (from phase2_step2_freeze_metrics_summary.txt)
**Model**: TCN [32, 32, 64], dilations [1, 2, 4], window=200
**Training**: 10 scenarios, MSE loss on residual

**Results**:
- Engine band (50-400Hz): **3.62 dB ± 1.24 dB**
- Road band (20-2000Hz): 3.01 dB ± 0.90 dB
- Cabin band (0-1000Hz): 2.81 dB ± 0.99 dB
- Full band: 2.76 dB ± 0.88 dB

**Gap to Goal**:
- Engine: Need +11.4 dB more
- Road: Need +7.0 dB more
- Cabin: Need +7.2 dB more

---

## Experiments

_Template for logging experiments:_

```
### Experiment [ID]: [Short Name]
**Date**: YYYY-MM-DD
**Hypothesis**: What we're testing and why
**Implementation**:
  - Model architecture: [details]
  - Training config: [details]
  - Data: [number of scenarios, any changes]
**Training Details**:
  - Epochs: X
  - Best val loss: X.XXe-XX
  - Training time: X minutes
**Results**:
  - Engine band: X.XX dB ± Y.YY dB (Δ from baseline: +/-Z.ZZ dB)
  - Road band: X.XX dB ± Y.YY dB
  - Cabin band: X.XX dB ± Y.YY dB
  - Full band: X.XX dB ± Y.YY dB
**Per-scenario breakdown**:
  - val_000: [results]
  - val_001: [results]
  - val_002: [results]
**Observations**:
  - [Qualitative findings from audio]
  - [Any anomalies or interesting patterns]
**Analysis**:
  - [Why did this work/not work?]
  - [What does this tell us about the problem?]
**Next Steps**:
  - [What to try next based on these results]
**Files**:
  - Model: phase_2_new/models/[experiment_id].keras
  - Metrics: phase_2_new/results/[experiment_id]/metrics.txt
  - Audio: phase_2_new/results/[experiment_id]/audio/
```

---

_Experiments will be logged below as they are conducted._

