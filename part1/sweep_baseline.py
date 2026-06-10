"""
Trains REINFORCE with several constant baseline values and plots them together to empirically justify the choice of baseline.

Baseline values swept: 0, 5, 10, 20, 50, 100, 200

Each run saves its own CSV under Logs/sweep/b_<value>/
A final comparison plot is saved to Logs/sweep/baseline_sweep.png
"""

import os
import csv
import time
import torch
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agent_reinforce_b_0 import Policy_Reinforce

SEED = 67   # best seed on Reinforce b=20

# Minimal inline agent that accepts any constant baseline 
def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add  = 0
    for t in reversed(range(r.size(-1))):
        running_add     = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class Agent_ConstBaseline:
    # REINFORCE with a fixed constant baseline b (subtracted from returns)
    def __init__(self, policy, baseline=0.0, device='cpu'):
        self.train_device  = device
        self.policy = policy.to(device)
        self.baseline = baseline
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.7)
        self.gamma = 0.99
        self.action_log_probs = []
        self.rewards = []

    def get_action(self, state):
        x = torch.from_numpy(state).float().to(self.train_device)
        dist = self.policy(x)
        a = dist.sample()
        return a, dist.log_prob(a).sum()

    def store_outcome(self, log_prob, reward):
        self.action_log_probs.append(log_prob)
        self.rewards.append(torch.tensor([reward]))

    def update_policy(self):
        log_probs = torch.stack(self.action_log_probs).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards).to(self.train_device).squeeze(-1)
        returns = discount_rewards(rewards, self.gamma)
        loss = (-(log_probs * (returns - self.baseline))).mean()
        # self.scheduler.step()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.action_log_probs, self.rewards = [], []
        return loss.item()


BASELINE_VALUES = [0, 5, 10, 20, 50, 100, 200]
N_EPISODES = 3000
CONVERGENCE_THR = 1000.0
SWEEP_LOG_DIR = "Logs/sweep"
SAVE_PLOT = f"{SWEEP_LOG_DIR}/baseline_sweep.png"

PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728", "#9467bd","#8c564b","#e377c2"]


def train_one(baseline, device):
    # Train one REINFORCE run with the given baseline. Returns (rewards_array, conv_ep)
    run_dir = f"{SWEEP_LOG_DIR}/b_{baseline}"
    os.makedirs(run_dir, exist_ok=True)
    csv_path = f"{run_dir}/training.csv"
    env  = gym.make('Hopper-v4')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    env.reset(seed=SEED)    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = Policy_Reinforce(obs_dim, act_dim)
    agent = Agent_ConstBaseline(policy, baseline=float(baseline), device=device)

    total_rewards = []
    conv_ep = None

    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['episode', 'reward', 'avg_reward_100', 'std_reward_100'])

    t0 = time.perf_counter()
    for ep in range(N_EPISODES):
        state, _ = env.reset()
        done = False
        ep_rew = 0.0

        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy())
            done = terminated or truncated
            agent.store_outcome(log_prob, reward)
            state  = next_state
            ep_rew += reward

        agent.update_policy()
        total_rewards.append(ep_rew)

        window = total_rewards[-100:]
        avg_reward = float(np.mean(window))
        std_reward = float(np.std(window))

        if conv_ep is None and avg_reward >= CONVERGENCE_THR:
            conv_ep = ep

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, ep_rew, avg_reward, std_reward])

        if (ep + 1) % 500 == 0:
            elapsed = (time.perf_counter() - t0) / 60
            print(f"  b={baseline:>4}  ep {ep+1:4d}/{N_EPISODES} | "
                  f"avg={avg_reward:7.1f} ± {std_reward:6.1f} | "
                  f"{elapsed:.1f}min")

    elapsed  = (time.perf_counter() - t0) / 60
    conv_str = str(conv_ep) if conv_ep else "Never"
    print(f"  b={baseline:>4}  DONE | final avg={avg_reward:.1f} | "
          f"converged={conv_str} | {elapsed:.1f}min")
    env.close()
    return np.array(total_rewards), conv_ep


def plot_sweep(results):
    
    # Single-panel plot: 100-ep rolling average reward per baseline value + ±1 std band
    
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for i, (b, (rewards, _)) in enumerate(sorted(results.items())):
        color = PALETTE[i % len(PALETTE)]
        roll_avg = np.convolve(rewards, np.ones(100) / 100, mode='valid')
        roll_std = np.array([rewards[max(0, j - 99):j + 1].std() for j in range(99, len(rewards))])
        x = np.arange(99, len(rewards))

        ax.plot(x, roll_avg, color=color, linewidth=1.8, label=f"b={b}")
        ax.fill_between(x, roll_avg - roll_std, roll_avg + roll_std,color=color, alpha=0.10)

    ax.set_title("Baseline Sweep – REINFORCE on Hopper-v4\n"
                 "(100-ep rolling average ± std)",fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Average Reward', fontsize=13)
    ax.grid(True, linestyle='--', linewidth=0.7, color='#cccccc', alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9, edgecolor='#cccccc')
    ax.set_xlim(left=0)
    ax.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    plt.savefig(SAVE_PLOT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSweep plot saved → {SAVE_PLOT}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Baseline sweep on {device}  |  baselines={BASELINE_VALUES}  |  episodes={N_EPISODES}\n")
    os.makedirs(SWEEP_LOG_DIR, exist_ok=True)

    results = {}
    for b in BASELINE_VALUES:
        print(f"\n── Training REINFORCE with b={b} ──────────────────────────────")
        rewards, conv_ep = train_one(b, device)
        results[b] = (rewards, conv_ep)

    plot_sweep(results)

    # Save summary to txt
    summary_path = f"{SWEEP_LOG_DIR}/sweep_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Baseline Sweep – REINFORCE on Hopper-v4\n")
        f.write(f"Seed: {SEED}  |  Episodes: {N_EPISODES}\n")
        f.write("=" * 46 + "\n")
        f.write(f"{'Baseline':>10} {'Conv. Episode':>14} {'Final Avg (100ep)':>18}\n")
        f.write("-" * 46 + "\n")
        for b in sorted(results):
            rewards, conv_ep = results[b]
            final_avg = float(np.mean(rewards[-100:]))
            conv_str = str(conv_ep) if conv_ep else "Never"
            line = f"{b:>10} {conv_str:>14} {final_avg:>18.1f}"
            f.write(line + "\n")
        f.write(f"\nNote: Conv. Episode = first ep where 100-ep avg >= {CONVERGENCE_THR}\n")

    print(f"Summary saved → {summary_path}")


if __name__ == '__main__':
    main()