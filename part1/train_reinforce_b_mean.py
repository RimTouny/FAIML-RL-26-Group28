"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""

import os
import csv
import time
import shutil
import torch
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agent_reinforce_b_mean import Agent_Mean, Policy_Mean

SEEDS = [20, 42, 67, 128]


def evaluate_policy(agent, env, n_eval_episodes=50, seed=0):

    eval_rewards = []
    for i in range(n_eval_episodes):
        state, _ = env.reset(seed=seed + i)
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            next_state, reward, terminated, truncated, _ = env.step(
                action.detach().cpu().numpy())
            done = terminated or truncated
            ep_reward += reward
            state = next_state
        eval_rewards.append(ep_reward)
    return float(np.mean(eval_rewards)), float(np.std(eval_rewards))

# Plotting helpers
def plot_reward_and_time(csv_path, save_path, title="REINFORCE – Mean Baseline"):
    """
    Two-panel figure:
      Top   : raw episode reward + 100-ep rolling average + ±1 std band
      Bottom: per-episode wall-clock time (seconds)
    """
    episodes, rewards, avgs, stds, ep_times = [], [], [], [], []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row['episode']))
            rewards.append(float(row['reward']))
            avgs.append(float(row['avg_reward_100']))
            stds.append(float(row['std_reward_100']))
            ep_times.append(float(row['episode_time_s']))

    avgs = np.array(avgs)
    stds = np.array(stds)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('white')

    # Top panel: reward
    ax1.set_facecolor('white')
    ax1.plot(episodes, rewards,color='#aec6e8', linewidth=0.6, alpha=0.7,label='Raw Episode Reward', zorder=1)
    ax1.plot(episodes, avgs,color='#0a1a6b', linewidth=2.2,label='100-Episode Average', zorder=2)
    # ±1 std shaded band — shows reward variance (spread = training stability)
    ax1.fill_between(episodes, avgs - stds, avgs + stds,color='#0a1a6b', alpha=0.12,label='±1 Std (100-ep)', zorder=0)
    ax1.set_title(title, fontsize=16, fontweight='bold', pad=12)
    ax1.set_ylabel('Total Reward', fontsize=13)
    ax1.grid(True, linestyle='--', linewidth=0.7, color='#cccccc', alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9, edgecolor='#cccccc')
    ax1.set_xlim(left=0)
    ax1.tick_params(axis='both', labelsize=11)

    # Bottom panel: time per episode
    ax2.set_facecolor('white')
    ax2.plot(episodes, ep_times,color='#e07b39', linewidth=0.8, alpha=0.85,label='Episode Time (s)')
    rolling_time = np.convolve(ep_times, np.ones(100) / 100, mode='valid')
    ax2.plot(range(99, len(ep_times)), rolling_time,color='#7a3010', linewidth=2.0, label='100-ep Avg Time')
    ax2.set_xlabel('Episode', fontsize=13)
    ax2.set_ylabel('Time (s)', fontsize=13)
    ax2.grid(True, linestyle='--', linewidth=0.7, color='#cccccc', alpha=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#cccccc')
    ax2.set_xlim(left=0)
    ax2.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved → {save_path}")


def run_seed(seed, device):
    # Run one full training session for the given seed. Returns (mean_reward, std_reward, model_path).
    print(f"\n{'─'*60}")
    print(f"  REINFORCE Mean Baseline | Seed {seed}")
    print(f"{'─'*60}")

    os.makedirs("Logs/mean", exist_ok=True)
    os.makedirs("models/mean", exist_ok=True)

    csv_path      = f"Logs/mean/training_mean_seed{seed}.csv"
    eval_csv_path = f"Logs/mean/eval_mean_seed{seed}.csv"
    model_path    = f"models/mean/model_mean_seed{seed}.pt"
    best_model_path = f"models/mean/model_mean_seed{seed}_best.pt"
    plot_path     = f"Logs/mean/learning_curve_baseline_mean_seed{seed}.png"

    env = gym.make('Hopper-v4')
    eval_env = gym.make('Hopper-v4')

    torch.manual_seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = Policy_Mean(obs_dim, act_dim)
    agent = Agent_Mean(policy, device=device)

    total_rewards = []
    episode_lengths = []
    n_episodes = 10000
    EVAL_INTERVAL = 250   # evaluate every N training episodes
    N_EVAL_EPS = 50    # deterministic test episodes per checkpoint

    best_eval_mean = -float('inf')   # tracks the highest eval mean seen so far

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "reward", "length",
            "avg_reward_100", "std_reward_100",
            "policy_loss", "entropy",
            "episode_time_s", "cumulative_time_s"
        ])

    with open(eval_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["training_episode", "eval_mean_return", "eval_std_return"])

    training_start = time.perf_counter()

    for ep in range(n_episodes):
        ep_start = time.perf_counter()

        done = False
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0

        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
            done = terminated or truncated

            agent.store_outcome(state, next_state, log_prob, reward, done)
            state = next_state
            episode_reward += reward
            episode_length += 1

        # Entropy estimate (before update clears buffers)
        entropies = [-lp.item() for lp in agent.action_log_probs]
        entropy = np.mean(entropies)

        policy_loss = agent.update_policy()

        ep_time = time.perf_counter() - ep_start
        cumulative_time = time.perf_counter() - training_start

        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Rolling statistics over last 100 episodes
        window = total_rewards[-100:]
        avg_reward = float(np.mean(window))
        std_reward = float(np.std(window))

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, episode_reward, episode_length,
                avg_reward, std_reward,
                policy_loss, entropy,
                round(ep_time, 4), round(cumulative_time, 2)
            ])

        # Evaluation checkpoint
        # Uses evaluation=True so the policy acts deterministically (mean action, no sampling noise)
        # The best-performing checkpoint is saved separately from the final model
        if (ep + 1) % EVAL_INTERVAL == 0:
            eval_mean, eval_std = evaluate_policy(agent, eval_env, n_eval_episodes=N_EVAL_EPS, seed=seed)
            with open(eval_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([ep + 1, eval_mean, eval_std])
            print(f"  [EVAL] ep {ep+1:5d} | eval_mean: {eval_mean:7.2f} ± {eval_std:6.2f}")
            if eval_mean > best_eval_mean:
                best_eval_mean = eval_mean
                torch.save(agent.policy.state_dict(), best_model_path)

        if (ep + 1) % 100 == 0:
            print(
                f"Episode {ep+1:4d}/{n_episodes} | "
                f"Reward: {episode_reward:7.2f} | "
                f"Len: {episode_length} | "
                f"Avg: {avg_reward:7.2f} ± {std_reward:6.2f} | "
                f"Loss: {policy_loss:8.2f} | "
                f"Entropy: {entropy:7.4f} | "
                f"EpTime: {ep_time:.4f}s | "
                f"Total: {cumulative_time:.2f}s"
            )

    torch.save(agent.policy.state_dict(), model_path)
    print(f"Model saved → {model_path}")
    print(f"Best eval model saved → {best_model_path}  (eval mean: {best_eval_mean:.2f})")

    plot_reward_and_time(csv_path, plot_path,title=f"Learning Curve: REINFORCE – Mean Baseline (Seed {seed})")

    # Final 100-episode average — reflects final policy quality, not full training avg
    mean_r = float(np.mean(total_rewards[-100:]))
    std_r = float(np.std(total_rewards[-100:]))

    return mean_r, std_r, best_eval_mean, model_path, best_model_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Mean Baseline REINFORCE on: {device}")
    print(f"Seeds: {SEEDS}")

    seed_stats = {}

    for seed in SEEDS:
        mean_r, std_r, best_eval_mean, model_path, best_model_path = run_seed(seed, device)
        seed_stats[seed] = (mean_r, std_r, best_eval_mean, model_path, best_model_path)
        print(f"\n  Seed {seed} summary → Mean(last100): {mean_r:.2f}  Std: {std_r:.2f}  Best Eval: {best_eval_mean:.2f}")

    print("\n" + "═"*60)
    print("  REINFORCE Mean Baseline — Results across seeds")
    print("═"*60)

    os.makedirs("Logs/mean", exist_ok=True)
    txt_path = "Logs/mean/seed_results.txt"

    with open(txt_path, "w") as f:
        f.write("Algorithm: REINFORCE – Mean Baseline\n")
        f.write("="*50 + "\n")
        for s in SEEDS:
            m, sd, best_eval, model_path, _ = seed_stats[s]
            # Read total time from last row of CSV
            csv_path = f"Logs/mean/training_mean_seed{s}.csv"
            with open(csv_path, 'r') as cf:
                last_row = list(csv.DictReader(cf))[-1]
                total_time = float(last_row['cumulative_time_s'])
            mins = total_time / 60
            line = f"Seed {s:3d}:  Mean = {m:8.2f}   Std = {sd:7.2f}   Best Eval = {best_eval:8.2f}   Time = {mins:.1f}min"
            print(line)
            f.write(line + "\n")

        # Best seed selected by best evaluation mean ,not training mean
        best_seed = max(seed_stats, key=lambda s: seed_stats[s][2])
        bm, bsd, beval, bmp, best_eval_mp = seed_stats[best_seed]
        summary = (f"\nBest seed : {best_seed}  "
                   f"(Mean = {bm:.2f}, Std = {bsd:.2f}, Best Eval = {beval:.2f})")
        print(summary)
        f.write(summary + "\n")

    print(f"\nResults saved → {txt_path}")

    # Save best model (chosen by eval performance, not training mean)
    best_model_dest = "models/mean/model_mean_best.pt"
    shutil.copy(best_eval_mp, best_model_dest)
    print(f"\n{'★'*60}")
    print(f"  Best seed : {best_seed}  |  Mean: {bm:.2f}  Std: {bsd:.2f}  Best Eval: {beval:.2f}")
    print(f"  Best model saved → {best_model_dest}")
    print(f"{'★'*60}")


if __name__ == '__main__':
    main()