"""
compare_plot.py
===============
Run after training scripts have finished.

Produces:
  Logs/comparison_all_configurations.png

The script compares SAC no-randomization source, SAC no-randomization target,
SAC UDR, and SAC ADR learning curves. This is intended for the final
domain-randomization comparison after SAC is selected as the better algorithm.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_udr_csv():
    canonical = "Logs/sac_push_udr_source_best/training_sac_push_udr_source_best.csv"
    if os.path.exists(canonical):
        return "SAC - UDR (best)", canonical

    candidates = glob.glob("Logs/sac_*_udr_*_source*/training_*.csv")
    if not candidates:
        return None, None

    def _final_avg(path):
        last = -1e9
        try:
            with open(path) as f:
                for row in csv.DictReader(f):
                    try:
                        last = float(row["avg_reward_100"])
                    except (KeyError, ValueError):
                        pass
        except OSError:
            pass
        return last

    best = max(candidates, key=_final_avg)

    try:
        parts = os.path.basename(os.path.dirname(best)).split("_udr_")
        lo, hi = parts[1].split("_")[:2]
        label = f"SAC - UDR [{lo}, {hi}]"
    except Exception:
        label = "SAC - UDR"

    return label, best


_udr_label, _udr_path = _find_udr_csv()

RUNS: dict[str, str] = {
    "SAC - no rand (source)": "Logs/sac_push_none_source_best/training_sac_push_none_source_best.csv",
    "SAC - no rand (target)": "Logs/sac_push_none_target_best/training_sac_push_none_target_best.csv",
    "SAC - ADR": "Logs/sac_push_adr_source_best/training_sac_push_adr_source_best.csv",
}

if _udr_path is not None:
    RUNS[_udr_label] = _udr_path

SAVE_PATH = "Logs/comparison_all_configurations.png"


def load_csv(path):
    episodes, avgs, stds, ep_times, success_rates = [], [], [], [], []

    with open(path) as f:
        for row in csv.DictReader(f):
            episodes.append(int(row["episode"]))
            avgs.append(float(row["avg_reward_100"]))
            stds.append(float(row["std_reward_100"]))
            ep_times.append(float(row["episode_time_s"]))
            success_rates.append(float(row["success_rate_100"]))

    return (np.array(episodes),np.array(avgs),np.array(stds),np.array(ep_times),np.array(success_rates),)


def rolling_mean(arr, w = 100):
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def main():
    os.makedirs("Logs", exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(13, 10),gridspec_kw={"height_ratios": [3, 1, 1]},)

    convergence_info = []

    for label, path in RUNS.items():
        try:
            eps, avgs, stds, ep_times, success_rates = load_csv(path)
        except FileNotFoundError:
            print(f"[WARN] CSV not found, skipping: {path}")
            continue

        ax1.plot(eps, avgs, linewidth=2.0, label=label)
        ax1.fill_between(eps, avgs - stds, avgs + stds, alpha=0.10)

        if len(ep_times) >= 100:
            roll = rolling_mean(ep_times, 100)
            ax2.plot(range(99, len(ep_times)), roll, linewidth=1.8, label=label)

        ax3.plot(eps, success_rates, linewidth=1.8, label=label)

        conv_mask = np.where(avgs >= -5.0)[0]
        conv_ep = int(eps[conv_mask[0]]) if len(conv_mask) else None
        convergence_info.append((label, conv_ep, float(avgs[-1]), float(stds[-1]), float(success_rates[-1])))

    ax1.set_title("Configuration Comparison - PandaPush-v3", fontsize=16, fontweight="bold")
    ax1.set_ylabel("100-episode avg return")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.set_ylabel("Time (s)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Success")
    ax3.set_ylim(0, 1)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Comparison plot saved -> {SAVE_PATH}")

    print("\nConvergence Summary")
    print("-" * 90)
    print(f"{'Configuration':<32} {'Conv. Ep':>10} {'Final Avg':>12} {'Final Std':>12} {'Final Suc':>12}")
    print("-" * 90)

    for label, conv_ep, final_avg, final_std, final_suc in convergence_info:
        conv_str = str(conv_ep) if conv_ep is not None else "Never"
        print(f"{label:<32} {conv_str:>10} {final_avg:>12.2f} {final_std:>12.2f} {final_suc:>12.1%}")


if __name__ == "__main__":
    main()