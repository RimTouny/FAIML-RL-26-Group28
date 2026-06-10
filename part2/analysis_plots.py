"""
analysis_plots.py
=================
Run after all training and evaluation scripts have finished.

Produces:
  Logs/domain_gap.png
  Logs/sensitivity_overlay.png
  Logs/udr_mass_histogram.png
  Logs/adr_evolution.png
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _scan_sensitivity_at(mass, keyword):
    for path in glob.glob("results/sensitivity_*.csv"):
        if keyword not in path:
            continue

        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    if abs(float(row["mass"]) - mass) < 0.01:
                        return float(row["success_rate"])
                except (KeyError, ValueError):
                    pass

    return None


def _final_success_from_training(csv_path):
    if not os.path.exists(csv_path):
        return None

    last = None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                last = float(row["success_rate_100"])
            except (KeyError, ValueError):
                pass

    return last


TRAINING_CSVS = {
    "none_source": "Logs/sac_push_none_source_best/training_sac_push_none_source_best.csv",
    "none_target": "Logs/sac_push_none_target_best/training_sac_push_none_target_best.csv",
    "udr_source": "Logs/sac_push_udr_source_best/training_sac_push_udr_source_best.csv",
    "adr_source": "Logs/sac_push_adr_source_best/training_sac_push_adr_source_best.csv",
}


def plot_domain_gap():
    configs = [
        ("Source->Source\nreference", "none_source", "source"),
        ("Source->Target\nlower bound", "none_source", "target"),
        ("UDR->Target", "udr_source", "target"),
        ("ADR->Target", "adr_source", "target"),
        ("Target->Target\nupper bound", "none_target", "target"),
    ]

    labels, source_vals, target_vals = [], [], []

    for label, key, eval_domain in configs:
        same_success = _final_success_from_training(TRAINING_CSVS.get(key, "")) or 0.0

        if eval_domain == "source":
            source_success = _scan_sensitivity_at(1.0, key) or same_success
            target_success = 0.0
        else:
            source_success = same_success if "source" in key else 0.0
            target_success = _scan_sensitivity_at(5.0, key) or 0.0

        labels.append(label)
        source_vals.append(source_success)
        target_vals.append(target_success)

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_source = ax.bar(x - w / 2, source_vals, w, label="Source domain / 1 kg")
    bars_target = ax.bar(x + w / 2, target_vals, w, label="Target domain / 5 kg")

    for bar in list(bars_source) + list(bars_target):
        height = bar.get_height()
        if height > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2,height + 0.01,f"{height:.0%}",ha="center",va="bottom",fontsize=9,)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.15)
    ax.set_title("Domain Gap Analysis - Success Rate")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    out = "Logs/domain_gap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {out}")


def _label_from_sensitivity_name(path):
    name = os.path.basename(path)
    name = name.replace("sensitivity_model_", "")
    name = name.replace("sensitivity_", "")
    name = name.replace(".csv", "")
    return name.replace("_", " ")


def plot_sensitivity_overlay():
    csvs = glob.glob("results/sensitivity_*.csv")

    if not csvs:
        print("[WARN] No sensitivity CSVs found - skipping sensitivity overlay.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for path in sorted(csvs):
        masses, means, stds, succs = [], [], [], []

        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    masses.append(float(row["mass"]))
                    means.append(float(row["mean_return"]))
                    stds.append(float(row["std_return"]))
                    succs.append(float(row["success_rate"]))
                except (KeyError, ValueError):
                    pass

        if not masses:
            continue

        masses = np.array(masses)
        means = np.array(means)
        stds = np.array(stds)
        succs = np.array(succs)

        label = _label_from_sensitivity_name(path)

        ax1.plot(masses, means, "o-", linewidth=2, label=label)
        ax1.fill_between(masses, means - stds, means + stds, alpha=0.12)

        ax2.plot(masses, succs, "o-", linewidth=2, label=label)

    for ax in (ax1, ax2):
        ax.axvline(x=1.0, linestyle=":", linewidth=1.2, label="Source mass / 1 kg")
        ax.axvline(x=5.0, linestyle=":", linewidth=1.2, label="Target mass / 5 kg")
        ax.set_xlabel("Cube mass (kg)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    ax1.set_ylabel("Mean return")
    ax1.set_title("Sensitivity: Mean Return vs Mass")

    ax2.set_ylabel("Success rate")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Sensitivity: Success Rate vs Mass")

    plt.tight_layout()
    out = "Logs/sensitivity_overlay.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {out}")


def plot_udr_histogram():
    real_csvs = glob.glob("results/udr_mass_samples_*.csv")

    if not real_csvs:
        print("[WARN] No UDR mass sample CSVs found - skipping UDR histogram.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for path in sorted(real_csvs):
        masses = []

        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    masses.append(float(row["mass"]))
                except (KeyError, ValueError):
                    pass

        if not masses:
            continue

        label = os.path.basename(path).replace("udr_mass_samples_", "").replace(".csv", "")
        ax.hist(masses, bins=40, alpha=0.55, label=label, edgecolor="white")

    ax.axvline(x=1.0, linestyle="--", linewidth=1.5, label="Source mass / 1 kg")
    ax.axvline(x=5.0, linestyle="--", linewidth=1.5, label="Target mass / 5 kg")
    ax.set_xlabel("Cube mass (kg)")
    ax.set_ylabel("Count")
    ax.set_title("UDR - Sampled Mass Distribution")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = "Logs/udr_mass_histogram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {out}")


def _find_adr_history():
    canonical = "results/adr_history.csv"
    if os.path.exists(canonical):
        return canonical

    candidates = glob.glob("results/adr_history_*.csv")
    if candidates:
        return sorted(candidates)[-1]

    return None


def plot_adr_evolution():
    path = _find_adr_history()

    if path is None:
        print("[WARN] No ADR history CSV found - skipping ADR evolution.")
        return

    episodes, centers, deltas = [], [], []

    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                episodes.append(int(row["episode"]))
                centers.append(float(row["center"]))
                deltas.append(float(row["delta"]))
            except (KeyError, ValueError):
                pass

    if not episodes:
        print("[WARN] ADR history CSV is empty - skipping ADR evolution.")
        return

    episodes = np.array(episodes, dtype=float)
    centers = np.array(centers)
    deltas = np.array(deltas)

    lows = np.clip(centers - deltas, 0.1, 10.0)
    highs = np.clip(centers + deltas, 0.1, 10.0)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(episodes, centers, linewidth=2.2, label="ADR center")
    ax.fill_between(episodes, lows, highs, alpha=0.18, label="ADR range")
    ax.axhline(y=1.0, linestyle="--", linewidth=1.4, label="Source mass / 1 kg")
    ax.axhline(y=5.0, linestyle="--", linewidth=1.4, label="Target mass / 5 kg")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cube mass (kg)")
    ax.set_title("ADR - Curriculum Evolution")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    out = "Logs/adr_evolution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved -> {out}")


if __name__ == "__main__":
    os.makedirs("Logs", exist_ok=True)

    plot_domain_gap()
    plot_sensitivity_overlay()
    plot_udr_histogram()
    plot_adr_evolution()
