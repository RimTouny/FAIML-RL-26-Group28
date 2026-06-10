"""
Run AFTER all four training scripts have finished.
Reads each algorithm's seed_results.txt to find the best seed,
then loads that seed's CSV to produce a single comparison figure with:
  - Top    : 100-ep average reward for all algorithms (+ ±1 std bands)
  - Bottom : 100-ep average episode time for all algorithms

Also saves a summary txt with reward, convergence, and total training time.
"""

import csv
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Map: label → (seed_results_txt, csv_path_template)
RUNS = {
    "REINFORCE b=0" : ("Logs/vanilla/seed_results.txt", "Logs/vanilla/training_vanilla_seed{seed}.csv"),
    "REINFORCE b=20" : ("Logs/baseline/seed_results.txt", "Logs/baseline/training_baseline_seed{seed}.csv"),
    "REINFORCE b=mean" : ("Logs/mean/seed_results.txt", "Logs/mean/training_mean_seed{seed}.csv"),
    "Actor-Critic" : ("Logs/actor_critic/seed_results.txt","Logs/actor_critic/training_ac_seed{seed}.csv"),
}

COLORS = {
    "REINFORCE b=0" : "#e05c5c",
    "REINFORCE b=20" : "#e07b39",
    "REINFORCE b=mean" : "#2ca05a",
    "Actor-Critic" : "#0a1a6b",
}

SAVE_PATH = "Logs/comparison_all_algorithms.png"
SUMMARY_PATH = "Logs/comparison_summary.txt"


def get_best_seed(txt_path):
    """Parse seed_results.txt and return the best seed integer."""
    with open(txt_path, 'r') as f:
        for line in f:
            m = re.search(r'Best seed\s*:\s*(\d+)', line)
            if m:
                return int(m.group(1))
    raise ValueError(f"Could not find 'Best seed' in {txt_path}")


def load_csv(path):   
    # Load CSV and return episodes, avg rewards, stds, episode times, and total training time.
    episodes, avgs, stds, ep_times, cum_times= [], [], [], [], []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row['episode']))
            avgs.append(float(row['avg_reward_100']))
            stds.append(float(row['std_reward_100']))
            ep_times.append(float(row['episode_time_s']))
            cum_times.append(float(row['cumulative_time_s']))

    total_time = cum_times[-1] if cum_times else 0.0
    return (np.array(episodes), np.array(avgs), np.array(stds),
            np.array(ep_times), total_time)


def rolling_mean(arr, w=100):
    return np.convolve(arr, np.ones(w) / w, mode='valid')

def format_time(seconds):
    """Format seconds into a readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9),gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('white')

    convergence_info = []

    for label, (txt_path, csv_template) in RUNS.items():
        # Resolve best seed → CSV path
        try:
            best_seed = get_best_seed(txt_path)
            csv_path = csv_template.format(seed=best_seed)
            eps, avgs, stds, ep_times, total_time = load_csv(csv_path)
            print(f"  {label:<22} → best seed {best_seed}  ({csv_path})")
        except FileNotFoundError as e:
            print(f"[WARN] Skipping {label}: {e}")
            continue

        color = COLORS[label]

        # Top panel — reward curve with a light std band behind it
        ax1.plot(eps, avgs, color=color, linewidth=2.0,label=f"{label} (seed {best_seed})", zorder=2)
        ax1.fill_between(eps, avgs - stds, avgs + stds, color=color, alpha=0.10, zorder=1)

        # Bottom panel — smooth out the noisy per-episode times with a rolling window
        if len(ep_times) >= 100:
            roll_t = rolling_mean(ep_times, 100)
            ax2.plot(range(99, len(ep_times)), roll_t,color=color, linewidth=1.8, label=f"{label} (seed {best_seed})")

        # Find the first episode where avg reward crossed 200      
        conv_mask = np.where(avgs >= 200)[0]
        conv_ep = int(eps[conv_mask[0]]) if len(conv_mask) else None
        convergence_info.append((label, best_seed, conv_ep, float(avgs[-1]), float(stds[-1]), total_time))

    # Top panel styling
    ax1.set_facecolor('white')
    ax1.set_title("Algorithm Comparison – Hopper-v4 (best seed per algorithm)",fontsize=16, fontweight='bold', pad=12)
    ax1.set_ylabel('100-ep Average Reward', fontsize=13)
    ax1.grid(True, linestyle='--', linewidth=0.7, color='#cccccc', alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9, edgecolor='#cccccc')
    ax1.set_xlim(left=0)
    ax1.tick_params(axis='both', labelsize=11)

    # Bottom panel styling
    ax2.set_facecolor('white')
    ax2.set_title("Episode Time Comparison (100-ep rolling avg)",fontsize=12, pad=6)
    ax2.set_xlabel('Episode', fontsize=13)
    ax2.set_ylabel('Time (s)', fontsize=13)
    ax2.grid(True, linestyle='--', linewidth=0.7, color='#cccccc', alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#cccccc')
    ax2.set_xlim(left=0)
    ax2.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved → {SAVE_PATH}")

    print("\n── Performance Summary ──────────────────────────────────────────────────────")
    print(f"{'Algorithm':<22} {'Seed':>6} {'Conv. Ep':>10} {'Final Avg':>10} {'Final Std':>10} {'Train Time':>12}")
    print("-" * 76)
    for label, seed, conv_ep, final_avg, final_std, total_time in convergence_info:
        conv_str = str(conv_ep) if conv_ep is not None else "Never"
        print(f"{label:<22} {seed:>6} {conv_str:>10} {final_avg:>10.1f} "
              f"{final_std:>10.1f} {format_time(total_time):>12}")

    # Save summary to txt file
    with open(SUMMARY_PATH, 'w') as f:
        f.write("Algorithm Comparison – Hopper-v4\n")
        f.write("=" * 76 + "\n")
        f.write(f"{'Algorithm':<22} {'Seed':>6} {'Conv. Ep':>10} "
                f"{'Final Avg':>10} {'Final Std':>10} {'Train Time':>12}\n")
        f.write("-" * 76 + "\n")
        for label, seed, conv_ep, final_avg, final_std, total_time in convergence_info:
            conv_str = str(conv_ep) if conv_ep is not None else "Never"
            f.write(f"{label:<22} {seed:>6} {conv_str:>10} {final_avg:>10.1f} "
                    f"{final_std:>10.1f} {format_time(total_time):>12}\n")
        f.write("\n")
        f.write("Notes:\n")
        f.write("  Conv. Ep   : First episode where 100-ep avg reward >= 200\n")
        f.write("  Final Avg  : 100-ep average reward at end of training\n")
        f.write("  Final Std  : 100-ep reward std at end of training\n")
        f.write("  Train Time : Total wall-clock training time for best seed\n")

    print(f"\nSummary saved → {SUMMARY_PATH}")


if __name__ == '__main__':
    main()