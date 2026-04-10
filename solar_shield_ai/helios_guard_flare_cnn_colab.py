# =============================================================================
# Helios Guard - 1D-CNN Solar Flare Predictor (TinyML / CubeSat)
# =============================================================================
# Google Colab: Runtime - Change runtime type - GPU (optional, speeds training)
# Paste this entire file into a Colab cell, or upload and run with:
#   exec(open("helios_guard_flare_cnn_colab.py").read())
# Local / Colab: put dummy_dataset.py in the same folder (synthetic data, no NASA files).
#
# Dependencies: TensorFlow 2.x, NumPy, scikit-learn, Matplotlib
# (Colab has these preinstalled; if needed: pip install tensorflow scikit-learn)
#
# Windows console: avoid non-ASCII in print() (use ASCII only below).
# =============================================================================

from __future__ import annotations

import os
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from dummy_dataset import DEFAULT_SEED as SEED
from dummy_dataset import DEFAULT_TOTAL_SEC as TOTAL_SEC
from dummy_dataset import N_CHANNELS, WINDOW_SEC, make_dummy_dataset, quick_dummy_dataset
from model_architecture import architecture_text, build_helios_guard_cnn

# Quieter TensorFlow logs (repeat runs / notebooks)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Train/val split
VAL_FRACTION = 0.2


def make_representative_dataset(X_train: np.ndarray, n_samples: int = 500):
    """TFLite expects a callable that returns a generator of [batch] arrays."""

    def _rep():
        rng = np.random.default_rng(SEED)
        n = min(n_samples, len(X_train))
        idx = rng.choice(len(X_train), size=n, replace=False)
        for i in idx:
            yield [X_train[i : i + 1].astype(np.float32)]

    return _rep


def _safe_train_val_split(X, y):
    """Stratified split when both classes have enough samples; else random split."""
    try:
        return train_test_split(X, y, test_size=VAL_FRACTION, random_state=SEED, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=VAL_FRACTION, random_state=SEED)


def main(quick: bool = False):
    """
    Train on the dummy synthetic dataset (no NASA files).

    Parameters
    ----------
    quick
        If True, use ~20k s of synthetic data for a fast smoke test.
    """
    # Fresh graph each run (Colab / script re-run: avoids layer name clashes and memory growth)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*reset_default_graph.*")
        tf.keras.backend.clear_session()
    plt.close("all")
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("TensorFlow:", tf.__version__)
    print("Helios Guard: synthetic GOES + proton -> 1D-CNN (flare vs no-flare)")

    # --- Dummy synthetic data (GOES-like + proton; no NASA downloads)
    data = quick_dummy_dataset(seed=SEED) if quick else make_dummy_dataset(total_sec=TOTAL_SEC, window_sec=WINDOW_SEC, seed=SEED)
    X, y = data.X, data.y
    peak_indices = data.peak_indices
    print(f"Windows: {X.shape}, labels: {y.shape}, positive fraction: {y.mean():.3f}")

    X_train, X_val, y_train, y_val = _safe_train_val_split(X, y)

    # --- Model (see model_architecture.py)
    print(architecture_text())
    model = build_helios_guard_cnn()
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    # --- Train
    es = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=64,
        callbacks=[es],
        verbose=1,
    )

    # --- Validation predictions
    y_prob = model.predict(X_val, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(np.int32)
    y_true = y_val.astype(np.int32)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax_cm.figure.colorbar(im, ax=ax_cm)
    ax_cm.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Pred: no flare", "Pred: flare"],
        yticklabels=["True: no flare", "True: flare"],
        title="Helios Guard - confusion matrix (val)",
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

    # --- Prediction lead time (seconds before synthetic peak when alarm first fires)
    full_prob = model.predict(X, verbose=0).ravel()
    triggered = np.where(full_prob >= 0.5)[0]
    lead_times_sec: list[float] = []

    for peak in peak_indices:
        if triggered.size == 0:
            break
        end_times = triggered + WINDOW_SEC - 1
        before_peak = triggered[end_times <= peak]
        if before_peak.size == 0:
            continue
        first_i = int(before_peak.min())
        end_t = first_i + WINDOW_SEC - 1
        lead = float(peak - end_t)
        if lead >= 0:
            lead_times_sec.append(lead)

    fig_lt, ax_lt = plt.subplots(figsize=(7, 4))
    if lead_times_sec:
        ax_lt.hist(lead_times_sec, bins=min(20, max(5, len(lead_times_sec))), color="steelblue", edgecolor="black", alpha=0.85)
        ax_lt.axvline(np.mean(lead_times_sec), color="darkorange", linestyle="--", label=f"Mean: {np.mean(lead_times_sec):.1f} s")
        ax_lt.set_title("Helios Guard - prediction lead time (s before peak, first P>=0.5)")
        ax_lt.set_xlabel("Seconds before flare peak (first P>=0.5 window)")
        ax_lt.set_ylabel("Count")
        ax_lt.legend()
    else:
        ax_lt.text(0.5, 0.5, "No lead times (no pre-peak triggers)", ha="center", va="center", transform=ax_lt.transAxes)
        ax_lt.set_title("Prediction lead time")
    plt.tight_layout()

    # --- INT8 TFLite export (ARM Cortex-M style)
    export_path = os.path.join(tempfile.gettempdir(), "helios_guard_flare_int8.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(X_train)

    tflite_model: bytes
    int8_io = True
    try:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
    except Exception as e:
        print("INT8-only conversion failed; falling back to dynamic-range quantization:", e)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = make_representative_dataset(X_train)
        tflite_model = converter.convert()
        int8_io = False
        export_path = os.path.join(tempfile.gettempdir(), "helios_guard_flare_dynamic.tflite")

    with open(export_path, "wb") as f:
        f.write(tflite_model)

    print(f"\nTFLite model written to: {export_path} (INT8 I/O: {int8_io})")
    print(f"Size: {os.path.getsize(export_path) / 1024:.2f} KB")

    # Sanity-check TFLite on one window (suppress deprecated Interpreter warning on TF 2.20+)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        interp = tf.lite.Interpreter(model_path=export_path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    sample = X_train[:1].astype(np.float32)
    if in_det["dtype"] == np.float32:
        interp.set_tensor(in_det["index"], sample)
    else:
        scale, zp = in_det["quantization"]
        if scale == 0:
            interp.set_tensor(in_det["index"], sample)
        else:
            sample_q = np.clip(np.round(sample / scale + zp), -128, 127).astype(np.int8)
            interp.set_tensor(in_det["index"], sample_q)
    interp.invoke()
    raw_out = np.array(interp.get_tensor(out_det["index"]))
    oscale, ozp = out_det["quantization"]
    fs = float(oscale) if oscale is not None else 0.0
    if fs != 0.0:
        prob_dequant = fs * (raw_out.astype(np.float32) - float(ozp))
        print("TFLite sample output (dequantized prob ~):", prob_dequant.ravel())
    else:
        print("TFLite sample output:", raw_out)

    plt.show()

    return model, history, cm, export_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Helios Guard 1D-CNN (dummy GOES/proton data)")
    p.add_argument("--quick", action="store_true", help="Short synthetic series for faster testing")
    args = p.parse_args()
    main(quick=args.quick)
