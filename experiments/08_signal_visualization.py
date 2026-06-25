"""
Signal-level visualizations for the SHU motor imagery EEG project.

This script complements the classification experiments with report-ready
signal analysis figures:

  1. C3/C4 relative 8-30 Hz power dynamics, used as an ERD/ERS-style figure.
  2. Welch PSD curves on C3/C4 for left-hand and right-hand MI.
  3. Mean band power on C3/C4 for mu, beta and mu+beta bands.

Important note:
  The MAT files contain the 4-second MI segment (1000 samples at 250 Hz).
  Therefore the ERD/ERS-style curve uses the first 0.5 s of this available
  segment as an internal baseline. In the report, describe it as relative
  band-power dynamics or ERD/ERS-style visualization, not as a full
  cue-locked rest-to-imagery analysis using the original 0-8 s trial.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAT_DIR = PROJECT_ROOT / "data" / "raw" / "mat"
METADATA_DIR = PROJECT_ROOT / "data" / "raw" / "metadata"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"
RESULTS_LOGS = PROJECT_ROOT / "results" / "logs"

FS = 250.0
N_SAMPLES = 1000
BASELINE_SAMPLES = int(0.5 * FS)
LABEL_NAMES = {1: "left", 2: "right"}


@dataclass
class BandPowerRow:
    label: str
    channel: str
    band: str
    mean_power: float
    log10_mean_power: float
    n_trials: int


@dataclass
class RelativePowerRow:
    label: str
    channel: str
    band: str
    baseline_power: float
    mean_relative_percent: float
    min_relative_percent: float
    max_relative_percent: float
    mean_relative_db: float
    min_relative_db: float
    max_relative_db: float
    n_trials: int


def read_channel_names() -> list[str]:
    path = METADATA_DIR / "task-motorimagery_channels.tsv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [row["name"].upper() for row in reader]


def find_channel_indices(channel_names: list[str]) -> dict[str, int]:
    required = ["C3", "C4"]
    missing = [name for name in required if name not in channel_names]
    if missing:
        raise ValueError(f"Missing required channels in metadata: {missing}")
    return {name: channel_names.index(name) for name in required}


def parse_subjects(args: argparse.Namespace) -> list[str]:
    if args.all_subjects:
        return [f"sub-{idx:03d}" for idx in range(1, 26)]
    return args.subjects


def list_mat_files(subjects: list[str]) -> list[Path]:
    paths: list[Path] = []
    for subject in subjects:
        paths.extend(sorted(MAT_DIR.glob(f"{subject}_ses-*_task_motorimagery_eeg.mat")))
    if not paths:
        raise FileNotFoundError(f"No MAT files found for subjects={subjects}")
    return sorted(paths)


def load_mat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mat = loadmat(path)
    x = np.asarray(mat["data"], dtype=np.float64)
    y = np.asarray(mat["labels"]).reshape(-1).astype(int)
    if x.ndim != 3:
        raise ValueError(f"{path.name}: expected data shape (trials, channels, samples), got {x.shape}")
    if x.shape[-1] != N_SAMPLES:
        raise ValueError(f"{path.name}: expected {N_SAMPLES} samples, got {x.shape[-1]}")
    # Remove per-trial/channel DC offset before spectral analysis.
    x = x - x.mean(axis=-1, keepdims=True)
    return x, y


def bandpass_sos(low: float, high: float, order: int = 4) -> np.ndarray:
    return signal.butter(order, [low, high], btype="bandpass", fs=FS, output="sos")


def smooth_curve(values: np.ndarray, window: int = 41) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def save_figure(filename: str) -> Path:
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    output = RESULTS_FIGURES / filename
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")
    return output


def write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames = list(rows[0].__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


class SignalAccumulator:
    def __init__(self, channel_indices: dict[str, int]):
        self.channel_indices = channel_indices
        self.channel_names = list(channel_indices.keys())
        self.target_indices = [channel_indices[name] for name in self.channel_names]

        self.erd_sos = bandpass_sos(8, 30)
        self.band_filters = {
            "mu_8_13Hz": bandpass_sos(8, 13),
            "beta_13_30Hz": bandpass_sos(13, 30),
            "mu_beta_8_30Hz": self.erd_sos,
        }

        self.power_sum = {
            label: np.zeros((len(self.channel_names), N_SAMPLES), dtype=np.float64)
            for label in LABEL_NAMES
        }
        self.power_count = {label: 0 for label in LABEL_NAMES}

        self.band_power_sum = {
            label: {
                band: np.zeros(len(self.channel_names), dtype=np.float64)
                for band in self.band_filters
            }
            for label in LABEL_NAMES
        }
        self.band_power_count = {label: 0 for label in LABEL_NAMES}

        self.psd_freqs: np.ndarray | None = None
        self.psd_sum = {
            label: None
            for label in LABEL_NAMES
        }
        self.psd_count = {label: 0 for label in LABEL_NAMES}

        self.files_seen = 0
        self.trials_seen = 0
        self.label_counts = {label: 0 for label in LABEL_NAMES}

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        self.files_seen += 1
        self.trials_seen += int(len(y))

        target = x[:, self.target_indices, :]

        # ERD/ERS-style 8-30 Hz relative power dynamics.
        erd_filtered = signal.sosfiltfilt(self.erd_sos, target, axis=-1)
        erd_power = erd_filtered * erd_filtered

        # Welch PSD on C3/C4.
        freqs, psd = signal.welch(target, fs=FS, nperseg=500, noverlap=250, axis=-1)
        psd_mask = (freqs >= 4) & (freqs <= 40)
        freqs = freqs[psd_mask]
        psd = psd[:, :, psd_mask]
        if self.psd_freqs is None:
            self.psd_freqs = freqs

        for label in LABEL_NAMES:
            idx = np.where(y == label)[0]
            if len(idx) == 0:
                continue
            self.label_counts[label] += int(len(idx))

            self.power_sum[label] += erd_power[idx].sum(axis=0)
            self.power_count[label] += int(len(idx))

            if self.psd_sum[label] is None:
                self.psd_sum[label] = np.zeros((len(self.channel_names), len(freqs)), dtype=np.float64)
            self.psd_sum[label] += psd[idx].sum(axis=0)
            self.psd_count[label] += int(len(idx))

            label_data = target[idx]
            for band, sos in self.band_filters.items():
                filtered = signal.sosfiltfilt(sos, label_data, axis=-1)
                band_power = np.mean(filtered * filtered, axis=-1)
                self.band_power_sum[label][band] += band_power.sum(axis=0)
            self.band_power_count[label] += int(len(idx))

    def mean_power_curves(self) -> dict[int, np.ndarray]:
        curves = {}
        for label in LABEL_NAMES:
            if self.power_count[label] == 0:
                raise ValueError(f"No trials for label {label}")
            curves[label] = self.power_sum[label] / self.power_count[label]
        return curves

    def relative_power_curves(self) -> dict[int, np.ndarray]:
        out = {}
        for label, curve in self.mean_power_curves().items():
            baseline = curve[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
            out[label] = (curve - baseline) / np.maximum(baseline, 1e-12) * 100.0
        return out

    def relative_power_db_curves(self) -> dict[int, np.ndarray]:
        out = {}
        for label, curve in self.mean_power_curves().items():
            baseline = curve[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
            out[label] = 10.0 * np.log10(np.maximum(curve, 1e-12) / np.maximum(baseline, 1e-12))
        return out

    def make_band_power_rows(self) -> list[BandPowerRow]:
        rows: list[BandPowerRow] = []
        for label, label_name in LABEL_NAMES.items():
            count = self.band_power_count[label]
            for band in self.band_filters:
                means = self.band_power_sum[label][band] / max(count, 1)
                for channel, value in zip(self.channel_names, means):
                    rows.append(
                        BandPowerRow(
                            label=label_name,
                            channel=channel,
                            band=band,
                            mean_power=float(value),
                            log10_mean_power=float(np.log10(max(value, 1e-12))),
                            n_trials=count,
                        )
                    )
        return rows

    def make_relative_power_rows(self) -> list[RelativePowerRow]:
        rows: list[RelativePowerRow] = []
        relative = self.relative_power_curves()
        relative_db = self.relative_power_db_curves()
        mean_power = self.mean_power_curves()
        for label, label_name in LABEL_NAMES.items():
            baseline = mean_power[label][:, :BASELINE_SAMPLES].mean(axis=1)
            for channel_index, channel in enumerate(self.channel_names):
                curve = relative[label][channel_index]
                curve_db = relative_db[label][channel_index]
                rows.append(
                    RelativePowerRow(
                        label=label_name,
                        channel=channel,
                        band="mu_beta_8_30Hz",
                        baseline_power=float(baseline[channel_index]),
                        mean_relative_percent=float(curve.mean()),
                        min_relative_percent=float(curve.min()),
                        max_relative_percent=float(curve.max()),
                        mean_relative_db=float(curve_db.mean()),
                        min_relative_db=float(curve_db.min()),
                        max_relative_db=float(curve_db.max()),
                        n_trials=self.power_count[label],
                    )
                )
        return rows


def plot_relative_power(acc: SignalAccumulator, tag: str) -> None:
    relative = acc.relative_power_db_curves()
    t = np.arange(N_SAMPLES) / FS
    colors = {1: "#4C78A8", 2: "#E45756"}

    plt.figure(figsize=(10.5, 6.4))
    for channel_index, channel in enumerate(acc.channel_names):
        plt.subplot(len(acc.channel_names), 1, channel_index + 1)
        for label, label_name in LABEL_NAMES.items():
            curve = smooth_curve(relative[label][channel_index], window=41)
            plt.plot(t, curve, label=label_name, color=colors[label], linewidth=1.6)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel(f"{channel}\nRel. power (dB)")
        plt.xlim(0, 4)
        if channel_index == 0:
            plt.title("C3/C4 relative 8-30 Hz power dynamics (ERD/ERS-style)")
            plt.legend(frameon=False, loc="upper right")
        if channel_index == len(acc.channel_names) - 1:
            plt.xlabel("Time within available 4-s MI segment (s)")
    save_figure(f"erd_ers_style_c3_c4_{tag}.png")


def plot_psd(acc: SignalAccumulator, tag: str) -> None:
    if acc.psd_freqs is None:
        raise RuntimeError("PSD has not been accumulated.")
    colors = {1: "#4C78A8", 2: "#E45756"}

    plt.figure(figsize=(10.5, 6.4))
    for channel_index, channel in enumerate(acc.channel_names):
        plt.subplot(len(acc.channel_names), 1, channel_index + 1)
        for label, label_name in LABEL_NAMES.items():
            psd_mean = acc.psd_sum[label][channel_index] / max(acc.psd_count[label], 1)
            plt.semilogy(acc.psd_freqs, psd_mean, label=label_name, color=colors[label], linewidth=1.6)
        plt.axvspan(8, 13, color="#F58518", alpha=0.12, label="mu (8-13 Hz)" if channel_index == 0 else None)
        plt.axvspan(13, 30, color="#54A24B", alpha=0.10, label="beta (13-30 Hz)" if channel_index == 0 else None)
        plt.ylabel(f"{channel}\nPSD")
        plt.xlim(4, 40)
        if channel_index == 0:
            plt.title("Welch PSD on C3/C4")
            plt.legend(frameon=False, loc="upper right", ncol=2)
        if channel_index == len(acc.channel_names) - 1:
            plt.xlabel("Frequency (Hz)")
    save_figure(f"psd_c3_c4_{tag}.png")


def plot_band_power(rows: list[BandPowerRow], tag: str) -> None:
    bands = ["mu_8_13Hz", "beta_13_30Hz", "mu_beta_8_30Hz"]
    channels = ["C3", "C4"]
    labels = ["left", "right"]
    row_map = {(row.label, row.channel, row.band): row for row in rows}

    x_labels = [band.replace("_", "\n") for band in bands]
    x = np.arange(len(bands))
    width = 0.36
    colors = {"left": "#4C78A8", "right": "#E45756"}

    plt.figure(figsize=(11.0, 4.8))
    for channel_index, channel in enumerate(channels):
        plt.subplot(1, 2, channel_index + 1)
        for label_index, label in enumerate(labels):
            values = [
                row_map[(label, channel, band)].log10_mean_power
                for band in bands
            ]
            xpos = x + (label_index - 0.5) * width
            plt.bar(xpos, values, width=width, color=colors[label], label=label)
        plt.xticks(x, x_labels)
        plt.ylabel("log10 mean band power")
        plt.title(f"{channel} band power")
        if channel_index == 0:
            plt.legend(frameon=False)
    save_figure(f"band_power_c3_c4_{tag}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=["sub-001"])
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--artifact-threshold",
        type=float,
        default=None,
        help="Optional visualization-only trial rejection threshold, applied to max abs amplitude across all channels.",
    )
    parser.add_argument(
        "--artifact-scope",
        choices=["target", "all"],
        default="target",
        help="Channels used for visualization-only artifact rejection when --artifact-threshold is set.",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    subjects = parse_subjects(args)
    paths = list_mat_files(subjects)
    tag_subject = "all_subjects" if args.all_subjects else "_".join(subjects)
    tag = f"{args.mode}_{tag_subject}"
    if args.artifact_threshold is not None:
        threshold_tag = f"{args.artifact_threshold:g}".replace(".", "p")
        tag = f"{tag}_clean_{args.artifact_scope}_abs{threshold_tag}"

    channel_names = read_channel_names()
    channel_indices = find_channel_indices(channel_names)
    acc = SignalAccumulator(channel_indices)
    raw_trials = 0
    rejected_trials = 0
    high_amplitude_trials_abs_gt_100 = 0
    source_files = len(paths)

    for path in tqdm(paths, desc="Signal visualization"):
        x, y = load_mat(path)
        raw_trials += int(len(y))
        max_abs_all = np.max(np.abs(x), axis=(1, 2))
        max_abs_target = np.max(np.abs(x[:, acc.target_indices, :]), axis=(1, 2))
        high_amplitude_trials_abs_gt_100 += int(np.sum(max_abs_all > 100.0))
        if args.artifact_threshold is not None:
            max_abs = max_abs_target if args.artifact_scope == "target" else max_abs_all
            keep = max_abs <= args.artifact_threshold
            rejected_trials += int(np.sum(~keep))
            x = x[keep]
            y = y[keep]
            if len(y) == 0:
                continue
        acc.update(x, y)

    band_rows = acc.make_band_power_rows()
    relative_rows = acc.make_relative_power_rows()

    band_csv = RESULTS_TABLES / f"signal_band_power_{tag}.csv"
    relative_csv = RESULTS_TABLES / f"signal_relative_power_{tag}.csv"
    write_csv(band_csv, band_rows)
    write_csv(relative_csv, relative_rows)

    plot_relative_power(acc, tag)
    plot_psd(acc, tag)
    plot_band_power(band_rows, tag)

    log = {
        "mode": args.mode,
        "subjects": subjects,
        "source_files": source_files,
        "files_used_after_visualization_rejection": acc.files_seen,
        "raw_trials_before_visualization_rejection": raw_trials,
        "n_trials_used": acc.trials_seen,
        "artifact_threshold": args.artifact_threshold,
        "artifact_scope": args.artifact_scope if args.artifact_threshold is not None else None,
        "rejected_trials_by_threshold": rejected_trials,
        "label_counts": {LABEL_NAMES[label]: count for label, count in acc.label_counts.items()},
        "channels": acc.channel_names,
        "channel_indices_0_based": channel_indices,
        "sampling_frequency": FS,
        "baseline_seconds_for_relative_power": BASELINE_SAMPLES / FS,
        "high_amplitude_trials_abs_gt_100_before_rejection": high_amplitude_trials_abs_gt_100,
        "note": (
            "MAT files contain the available 4-second MI segment. Relative power "
            "uses the first 0.5 s of this segment as an internal baseline. "
            "Artifact threshold, when provided, is used only for visualization."
        ),
    }
    RESULTS_LOGS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_LOGS / f"signal_visualization_{tag}.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"Saved: {band_csv}")
    print(f"Saved: {relative_csv}")
    print(f"Saved: {log_path}")
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
