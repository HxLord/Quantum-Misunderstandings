"""
Helios Guard — dummy / synthetic dataset (NumPy only, no NASA downloads).

Mimics GOES soft X-ray (0.1–0.8 nm) and proton-flux time series at 1 Hz for
testing the 1D-CNN pipeline immediately.

Usage:
    from dummy_dataset import make_dummy_dataset, quick_dummy_dataset

    data = make_dummy_dataset(total_sec=120_000, seed=42)
    # data.X: (N, 60, 2), data.y: (N,), data.peak_indices, ...

    small = quick_dummy_dataset()  # fast smoke test (~5k s)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

DEFAULT_SEED = 42
DEFAULT_TOTAL_SEC = 120_000
SAMPLING_HZ = 1
WINDOW_SEC = 60
N_CHANNELS = 2

QUIET_XRAY = 1e-6
PRE_FLARE_LEAD_SEC = 300
POST_FLARE_SEC = 120


class DummyDataset(NamedTuple):
    """Ready-to-train tensors + raw series for debugging / lead-time plots."""

    X: np.ndarray  # (N, window, 2) float32
    y: np.ndarray  # (N,) float32 {0,1}
    peak_indices: list[int]
    flare_mask: np.ndarray  # (T,) int32
    x_ray: np.ndarray  # (T,) raw synthetic X-ray proxy
    proton: np.ndarray  # (T,) raw synthetic proton proxy
    features_2d: np.ndarray  # (T, 2) log-normalized features
    window_sec: int
    sampling_hz: int


@dataclass(frozen=True)
class DummyDatasetConfig:
    total_sec: int = DEFAULT_TOTAL_SEC
    window_sec: int = WINDOW_SEC
    seed: int = DEFAULT_SEED
    quiet_xray: float = QUIET_XRAY
    pre_flare_lead_sec: int = PRE_FLARE_LEAD_SEC
    post_flare_sec: int = POST_FLARE_SEC


def generate_synthetic_goes_proton_series(
    total_sec: int,
    seed: int = DEFAULT_SEED,
    quiet_xray: float = QUIET_XRAY,
    pre_flare_lead_sec: int = PRE_FLARE_LEAD_SEC,
    post_flare_sec: int = POST_FLARE_SEC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Synthetic GOES-like X-ray and proton flux (no external files).

    Returns
    -------
    x_ray, proton : (T,) float64
    flare_mask : (T,) int32 — 1 inside labeled flare / pre-flare intervals
    peak_indices : sorted list of synthetic flare peak sample indices
    """
    rng = np.random.default_rng(seed)
    t = np.arange(total_sec, dtype=np.float64)

    x_ray = quiet_xray * (1.0 + 0.3 * np.sin(2 * np.pi * t / 86_400.0))
    x_ray *= rng.lognormal(mean=0.0, sigma=0.15, size=total_sec)

    proton = 1.0 + 0.5 * np.sin(2 * np.pi * t / (6 * 3600.0))
    proton *= rng.lognormal(mean=0.0, sigma=0.2, size=total_sec)

    flare_mask = np.zeros(total_sec, dtype=np.int32)
    peak_indices: list[int] = []

    n_flares = max(8, total_sec // 15_000)
    for _ in range(n_flares):
        peak = int(rng.integers(8_000, total_sec - 8_000))
        peak_indices.append(peak)

        amplitude = rng.uniform(50.0, 200.0)
        width_rise = rng.integers(40, 120)
        width_decay = rng.integers(200, 600)

        rel = t - peak
        rise = np.exp(np.minimum(rel / width_rise, 12.0))
        rise = np.where(rel < 0, rise, 1.0)
        decay = np.exp(-np.maximum(rel, 0.0) / width_decay)
        profile = amplitude * rise * decay
        x_ray += quiet_xray * profile

        proton_bump = 0.6 * amplitude * np.exp(-np.maximum(rel - 60.0, 0.0) / (1.5 * width_decay))
        proton_bump *= np.exp(np.minimum((rel - 30.0) / max(width_rise, 1), 8.0))
        proton_bump = np.where(rel < -200, 0.0, proton_bump)
        proton += proton_bump

        t0 = max(0, peak - pre_flare_lead_sec)
        t1 = min(total_sec - 1, peak + post_flare_sec)
        flare_mask[t0 : t1 + 1] = 1

    return x_ray.astype(np.float64), proton.astype(np.float64), flare_mask, sorted(peak_indices)


def log_normalize_channels(x_ray: np.ndarray, proton: np.ndarray) -> np.ndarray:
    """log1p per channel, then per-channel z-score over time."""
    eps = 1e-12
    lx = np.log1p(np.maximum(x_ray, eps))
    lp = np.log1p(np.maximum(proton, eps))
    stacked = np.stack([lx, lp], axis=-1)
    mean = stacked.mean(axis=0, keepdims=True)
    std = stacked.std(axis=0, keepdims=True) + 1e-6
    return (stacked - mean) / std


def build_sliding_windows(
    features_2d: np.ndarray,
    flare_mask: np.ndarray,
    window: int,
    n_channels: int = N_CHANNELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (N, window, n_channels) and binary labels (flare present in window)."""
    t_total = features_2d.shape[0]
    n = t_total - window + 1
    if n <= 0:
        raise ValueError("Time series too short for the chosen window.")

    X = np.zeros((n, window, n_channels), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    for i in range(n):
        X[i] = features_2d[i : i + window].astype(np.float32)
        y[i] = 1.0 if np.any(flare_mask[i : i + window] == 1) else 0.0

    return X, y


def make_dummy_dataset(
    total_sec: int | None = None,
    window_sec: int | None = None,
    seed: int | None = None,
    config: DummyDatasetConfig | None = None,
) -> DummyDataset:
    """
    One-shot synthetic dataset for Helios Guard testing (no downloads).

    Parameters
    ----------
    total_sec
        Length of the series in seconds (1 Hz). Default 120_000 or ``config.total_sec``.
    window_sec
        Sliding window length. Default ``WINDOW_SEC`` or ``config.window_sec``.
    seed
        RNG seed for reproducible flares. Default ``DEFAULT_SEED`` or ``config.seed``.
    config
        Optional defaults bundle; explicit kwargs override config fields.
    """
    cfg = config or DummyDatasetConfig()
    total_sec = total_sec if total_sec is not None else cfg.total_sec
    window_sec = window_sec if window_sec is not None else cfg.window_sec
    seed = seed if seed is not None else cfg.seed
    qx = cfg.quiet_xray
    pre = cfg.pre_flare_lead_sec
    post = cfg.post_flare_sec

    x_ray, proton, flare_mask, peak_indices = generate_synthetic_goes_proton_series(
        total_sec, seed=seed, quiet_xray=qx, pre_flare_lead_sec=pre, post_flare_sec=post
    )
    features_2d = log_normalize_channels(x_ray, proton)
    X, y = build_sliding_windows(features_2d, flare_mask, window_sec, N_CHANNELS)

    return DummyDataset(
        X=X,
        y=y,
        peak_indices=peak_indices,
        flare_mask=flare_mask,
        x_ray=x_ray,
        proton=proton,
        features_2d=features_2d,
        window_sec=window_sec,
        sampling_hz=SAMPLING_HZ,
    )


def quick_dummy_dataset(seed: int = DEFAULT_SEED) -> DummyDataset:
    """Short series (~5.5 h) for fast pipeline checks."""
    return make_dummy_dataset(total_sec=20_000, window_sec=WINDOW_SEC, seed=seed)


def save_dummy_npz(path: str, data: DummyDataset) -> None:
    """Save arrays to disk for offline reuse (optional)."""
    np.savez_compressed(
        path,
        X=data.X,
        y=data.y,
        flare_mask=data.flare_mask,
        x_ray=data.x_ray,
        proton=data.proton,
        features_2d=data.features_2d,
        peak_indices=np.array(data.peak_indices, dtype=np.int64),
        window_sec=data.window_sec,
        sampling_hz=data.sampling_hz,
    )


def load_dummy_npz(path: str) -> DummyDataset:
    z = np.load(path, allow_pickle=False)
    peaks = z["peak_indices"].astype(np.int64).tolist()
    return DummyDataset(
        X=z["X"],
        y=z["y"],
        peak_indices=peaks,
        flare_mask=z["flare_mask"],
        x_ray=z["x_ray"],
        proton=z["proton"],
        features_2d=z["features_2d"],
        window_sec=int(z["window_sec"]),
        sampling_hz=int(z["sampling_hz"]),
    )


if __name__ == "__main__":
    d = quick_dummy_dataset()
    print("DummyDataset (quick):", d.X.shape, d.y.shape, "positives:", d.y.mean())
    print("Peaks:", len(d.peak_indices), "first few:", d.peak_indices[:5])
    out = "helios_guard_dummy_quick.npz"
    save_dummy_npz(out, d)
    print("Wrote", out)
