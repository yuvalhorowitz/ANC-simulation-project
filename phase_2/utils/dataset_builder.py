# phase2_new/utils/dataset_builder.py
"""
Phase 2 Dataset Builder

Loads:
  - *_ref.wav              (reference microphone)
  - *_err.wav              (error microphone)
  - *_rir_spk_to_err.npy   (secondary path h_s)

Builds:
  - sliding windows of ref mic (X)
  - aligned error targets (err)
  - secondary path impulse response (h_s)

This module contains NO ML code.
Its only responsibility is correct signal alignment.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import librosa


def load_scenario(
    scenario_prefix: str,
    fs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one Phase-2 scenario.

    Parameters
    ----------
    scenario_prefix : str
        Path prefix without suffix, e.g.:
        "data/train/train_scenario_000"
    fs : int
        Sampling rate

    Returns
    -------
    ref : np.ndarray
        Reference mic signal (1D)
    err : np.ndarray
        Error mic signal (1D)
    h_s : np.ndarray
        Secondary path impulse response (1D)
    """

    ref_path = scenario_prefix + "_ref.wav"
    err_path = scenario_prefix + "_err.wav"
    rir_path = scenario_prefix + "_rir_spk_to_err.npy"

    if not os.path.exists(ref_path):
        raise FileNotFoundError(ref_path)
    if not os.path.exists(err_path):
        raise FileNotFoundError(err_path)
    if not os.path.exists(rir_path):
        raise FileNotFoundError(rir_path)

    # Load audio
    ref, _ = librosa.load(ref_path, sr=fs, mono=True)
    err, _ = librosa.load(err_path, sr=fs, mono=True)

    # Load RIR
    h_s = np.load(rir_path)

    # Normalize (important for stable training)
    ref = ref / (np.max(np.abs(ref)) + 1e-9)
    err = err / (np.max(np.abs(err)) + 1e-9)
    h_s = h_s / (np.max(np.abs(h_s)) + 1e-9)

    return ref.astype(np.float32), err.astype(np.float32), h_s.astype(np.float32)


def build_windows(
    ref: np.ndarray,
    err: np.ndarray,
    window_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows from reference mic and aligned error targets.

    Alignment logic (CRITICAL):
    ---------------------------
    For each time index t:

      input  = ref[t-window_length : t]
      target = err[t]

    This preserves causality and matches FxLMS timing.

    Parameters
    ----------
    ref : np.ndarray
        Reference signal (1D)
    err : np.ndarray
        Error signal (1D)
    window_length : int
        Number of past samples used by controller

    Returns
    -------
    X : np.ndarray
        Shape (N, window_length, 1)
    err_t : np.ndarray
        Shape (N, 1)
    """

    if len(ref) != len(err):
        raise ValueError("ref and err must have same length")

    N = len(ref) - window_length
    if N <= 0:
        raise ValueError("Signal too short for given window_length")

    X = np.zeros((N, window_length, 1), dtype=np.float32)
    err_t = np.zeros((N, 1), dtype=np.float32)

    for i in range(N):
        t = i + window_length
        X[i, :, 0] = ref[t - window_length : t]
        err_t[i, 0] = err[t]

    return X, err_t


def build_training_example(
    scenario_prefix: str,
    fs: int,
    window_length: int,
):
    """
    High-level helper for Phase-2 training.

    Returns everything needed for training on ONE scenario.

    Returns
    -------
    X : np.ndarray
        (N, window_length, 1) reference windows
    err_t : np.ndarray
        (N, 1) error targets
    h_s : np.ndarray
        (K,) secondary path impulse response
    """

    ref, err, h_s = load_scenario(
        scenario_prefix=scenario_prefix,
        fs=fs,
    )

    X, err_t = build_windows(
        ref=ref,
        err=err,
        window_length=window_length,
    )

    return X, err_t, h_s


def list_scenarios(directory: str) -> list[str]:
    """
    List scenario prefixes in a directory.

    Example:
      data/train/train_scenario_000_ref.wav
      → data/train/train_scenario_000
    """

    prefixes = set()

    for fname in os.listdir(directory):
        if fname.endswith("_ref.wav"):
            prefix = fname.replace("_ref.wav", "")
            prefixes.add(os.path.join(directory, prefix))

    return sorted(prefixes)
