"""
Helios Guard: TinyML 1D-CNN for binary flare detection (GOES-like X-ray + proton window).

Input:  (batch, 60, 2)  -- 60 s at 1 Hz, 2 channels (log-scaled in the data pipeline).
Output: (batch, 1)      -- P(flare), sigmoid.

Layer stack (fixed for Cortex-M friendly export):
  Input
  Conv1D: 16 filters, kernel 3, padding same, ReLU
  MaxPooling1D: pool size 2
  Flatten
  Dense: 16 units, ReLU
  Dense: 1 unit, Sigmoid
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from dummy_dataset import N_CHANNELS, WINDOW_SEC


@dataclass(frozen=True)
class HeliosGuardCNNConfig:
    """Hyperparameters matching the Helios Guard TinyML specification."""

    window_samples: int = WINDOW_SEC
    n_channels: int = N_CHANNELS
    conv_filters: int = 16
    conv_kernel_size: int = 3
    pool_size: int = 2
    dense_units: int = 16
    dense_output: int = 1


DEFAULT_CONFIG = HeliosGuardCNNConfig()


def build_helios_guard_cnn(
    input_shape: tuple[int, int] | None = None,
    config: HeliosGuardCNNConfig = DEFAULT_CONFIG,
) -> tf.keras.Model:
    """
    Build the Keras Functional API model (no compilation).

    Parameters
    ----------
    input_shape
        (window, channels), default (60, 2).
    config
        Layer sizes; use DEFAULT_CONFIG for the standard Helios Guard stack.
    """
    if input_shape is None:
        input_shape = (config.window_samples, config.n_channels)

    inp = tf.keras.Input(shape=input_shape, name="goes_proton_window")
    x = tf.keras.layers.Conv1D(
        config.conv_filters,
        kernel_size=config.conv_kernel_size,
        padding="same",
        activation="relu",
        name="conv1d_16",
    )(inp)
    x = tf.keras.layers.MaxPooling1D(pool_size=config.pool_size, name="maxpool1d_2")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(config.dense_units, activation="relu", name="dense_16")(x)
    out = tf.keras.layers.Dense(config.dense_output, activation="sigmoid", name="flare_prob")(x)
    return tf.keras.Model(inp, out, name="helios_guard_flare_1dcnn")


def architecture_text(config: HeliosGuardCNNConfig = DEFAULT_CONFIG) -> str:
    """ASCII description for logs, Colab, or documentation."""
    w, c = config.window_samples, config.n_channels
    lines = [
        "Helios Guard 1D-CNN (TinyML)",
        "",
        f"Input:  ({w}, {c})  # 60 s x 2 ch (GOES X-ray proxy + proton proxy)",
        "",
        f"Conv1D(filters={config.conv_filters}, kernel={config.conv_kernel_size}, padding=same, activation=relu)",
        f"MaxPooling1D(pool_size={config.pool_size})",
        "Flatten",
        f"Dense({config.dense_units}, relu)",
        f"Dense({config.dense_output}, sigmoid)  # P(flare)",
    ]
    return "\n".join(lines)


def print_architecture(config: HeliosGuardCNNConfig = DEFAULT_CONFIG) -> None:
    """Print layer stack and Keras summary (standalone helper)."""
    print(architecture_text(config))
    build_helios_guard_cnn(config=config).summary()


if __name__ == "__main__":
    print_architecture()
