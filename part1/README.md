# Reinforcement Learning for Hopper-v4 – Policy Gradient Methods

This repository implements and compares four policy gradient algorithms on the MuJoCo **Hopper-v4** continuous control task. The goal is to train a policy from scratch using REINFORCE variants and an Actor-Critic method, and to systematically compare their learning speed, stability, and final performance.

---

## Repository Structure

```
part1/
├── agent_reinforce_b_0.py        # REINFORCE, no baseline
├── agent_reinforce_b_20.py       # REINFORCE, constant baseline b=20
├── agent_reinforce_b_mean.py     # REINFORCE, mean return baseline
├── agent_ac.py                   # Actor-Critic policy + agent
│
├── train_reinforce_b_0.py        # Training loop — vanilla REINFORCE
├── train_reinforce_b_20.py       # Training loop — constant baseline
├── train_reinforce_b_mean.py     # Training loop — mean baseline
├── train_ac.py                   # Training loop — Actor-Critic
│
├── sweep_baseline.py             # Sweeps b = 0,5,10,20,50,100,200 to pick best baseline
├── compare_plot.py               # Generates final 4-algorithm comparison figure
├── test_random_policy.py         # Environment exploration with a random policy
│
├── Logs/
│   ├── vanilla/                  # CSVs + plots for REINFORCE b=0 (one per seed)
│   │   └── seed_results.txt      # Mean/std/time per seed + best seed
│   ├── baseline/                 # CSVs + plots for REINFORCE b=20 (one per seed)
│   │   └── seed_results.txt
│   ├── mean/                     # CSVs + plots for REINFORCE b=mean (one per seed)
│   │   └── seed_results.txt
│   ├── actor_critic/             # CSVs + plots for Actor-Critic (one per seed)
│   │   └── seed_results.txt
│   ├── sweep/       
|   |   ├── b_0/training.csv
|   |   ├── b_5/training.csv
|   |   ├── b_10/training.csv
|   |   ├── b_20/training.csv
|   |   ├── b_50/training.csv
|   |   ├── b_100/training.csv
|   |   ├── b_200/training.csv
|   |   ├── baseline_sweep.png             # Per-baseline CSVs + sweep comparison plot
│   |   └── sweep_summary.txt  
│   ├── comparison_all_algorithms.png
│   └── comparison_summary.txt    # Final performance table across all algorithms
│
└── models/
    ├── vanilla/
    │   ├── model_vanilla_seed20.pt
    │   ├── model_vanilla_seed20_best.pt
    │   ├── model_vanilla_seed42.pt
    │   ├── model_vanilla_seed42_best.pt
    │   ├── model_vanilla_seed67.pt
    │   ├── model_vanilla_seed67_best.pt
    │   ├── model_vanilla_seed128.pt
    │   ├── model_vanilla_seed128_best.pt
    │   └── model_vanilla_best.pt
    │
    ├── baseline/
    │   ├── model_baseline_seed20.pt
    │   ├── model_baseline_seed20_best.pt
    │   ├── model_baseline_seed42.pt
    │   ├── model_baseline_seed42_best.pt
    │   ├── model_baseline_seed67.pt
    │   ├── model_baseline_seed67_best.pt
    │   ├── model_baseline_seed128.pt
    │   ├── model_baseline_seed128_best.pt
    │   └── model_baseline_best.pt
    │
    ├── mean/
    │   ├── model_mean_seed20.pt
    │   ├── model_mean_seed20_best.pt
    │   ├── model_mean_seed42.pt
    │   ├── model_mean_seed42_best.pt
    │   ├── model_mean_seed67.pt
    │   ├── model_mean_seed67_best.pt
    │   ├── model_mean_seed128.pt
    │   ├── model_mean_seed128_best.pt
    │   └── model_mean_best.pt
    │
    └── actor_critic/
        ├── model_ac_seed20.pt
        ├── model_ac_seed20_best.pt
        ├── model_ac_seed42.pt
        ├── model_ac_seed42_best.pt
        ├── model_ac_seed67.pt
        ├── model_ac_seed67_best.pt
        ├── model_ac_seed128.pt
        ├── model_ac_seed128_best.pt
        └── model_ac_best.pt
```

---
## Algorithms Implemented

| Algorithm    | Baseline        | Key detail                                        |
|--------------|-----------------|---------------------------------------------------|
| REINFORCE    | None (b=0)      | Pure Monte Carlo policy gradient                  |
| REINFORCE    | Constant (b=20) | Subtracts fixed scalar from returns               |
| REINFORCE    | Mean return     | Normalizes advantages with mean and std           |
| Actor-Critic | Learned V(s)    | TD bootstrapping, separate actor/critic networks  |
---

## Environment Setup
```bash
pip install torch gymnasium[mujoco] numpy matplotlib
```
MuJoCo binaries are included with `gymnasium[mujoco]` — no separate installation needed.

---
## Usage

**Step 1 — Explore the environment**
Run this first to inspect the state/action spaces and understand termination conditions:
```bash
python test_random_policy.py
```

**Step 2 — Train each algorithm**
Each script runs for 10,000 episodes across **4 seeds (20, 42, 67, 128)**. Per seed it saves a CSV log, a learning curve plot, and a model checkpoint. After all seeds finish it prints a mean/std/time summary table to the terminal, writes `seed_results.txt`, and copies the best-seed model to `model_<algo>_best.pt`.
```bash
python train_reinforce_b_0.py
python train_reinforce_b_20.py
python train_reinforce_b_mean.py
python train_ac.py
```
Or run all at once (PowerShell):
```powershell
python train_reinforce_b_0.py; python train_reinforce_b_20.py; python train_reinforce_b_mean.py; python train_ac.py; python compare_plot.py
```

**Step 3 — Baseline sweep**
Empirically compare constant baseline values b = 0, 5, 10, 20, 50, 100, 200 over 3,000 episodes each:
```bash
python sweep_baseline.py
```
Output goes to `Logs/sweep/`


**Step 4 — Generate comparison figure**
Run after all four training scripts have finished. Automatically picks the best seed per algorithm from `seed_results.txt`:
```bash
python compare_plot.py
```
Saves two files:
- `Logs/comparison_all_algorithms.png` — two-panel figure with reward curves and episode times
- `Logs/comparison_summary.txt` — performance table with reward, convergence, and training time

---
## Results

### Per-Algorithm Multi-Seed Training Results
Each `train_*.py` script runs the full 10,000-episode training loop for seeds **20, 42, 67, and 128** in sequence. All outputs are seed-stamped so runs never overwrite each other.

| Output                              | Description                                                                           |
|-------------------------------------|---------------------------------------------------------------------------------------|
| `training_<algo>_seed<N>.csv`       | Full episode log for seed N                                                           |
| `learning_curve_<algo>_seed<N>.png` | Two-panel learning curve for seed N                                                   |
| `model_<algo>_seed<N>.pt`           | Final policy weights after the full 10,000-episode run for seed N                     |
| `model_<algo>_seed<N>_best.pt`      | Best evaluation checkpoint for seed N, selected during periodic 50-episode evaluation |
| `seed_results.txt`                  | Mean ± std ± time per seed + best seed                                                |
| `model_<algo>_best.pt`              | Copy of the best-seed evaluation checkpoint across all seeds                          |

```
### Model Checkpoint Naming
For each seed, two model files may be saved:

| File                           | Meaning                                                                                   |
|--------------------------------|-------------------------------------------------------------------------------------------|
| `model_<algo>_seed<N>.pt`      | Final model at the end of the 10,000 training episodes                                    |
| `model_<algo>_seed<N>_best.pt` | Best checkpoint observed during periodic evaluation for that same seed                    |
| `model_<algo>_best.pt`         | Final selected model for the algorithm, copied from the best seed's `_best.pt` checkpoint |

```
`seed_results.txt` format :
-	Best seed selection — after all seeds finish training, the best seed per algorithm is chosen by its final deterministic evaluation score (mean over 50 episodes at end of training) rather than by training reward, ensuring the reported model reflects true generalization quality rather than a lucky training spike.

**Algorithm: REINFORCE – No Baseline (b=0)**
| Seed | Mean   | Std   | Best Eval | Time    |
|------|--------|-------|-----------|---------|
| 20   | 123.46 | 44.53 | 405.50    | 20.8min |
| 42   | 226.91 |  6.35 | 319.83    | 20.6min |
| 67   | 307.83 | 39.76 | 337.27    | 19.4min |
| 128  | 180.27 | 10.75 | 210.34    | 17.8min |
| **Best** | **Seed 20** | **Mean=123.46** | **Best Eval=405.50** |

**Algorithm: REINFORCE – b=20**
| Seed | Mean   | Std   | Best Eval | Time    |
|------|--------|-------|-----------|---------|
| 20   | 163.85 | 44.09 | 209.44    | 16.2min |
| 42   | 225.69 |  2.63 | 306.14    | 13.4min |
| 67   | 307.48 | 17.76 | 633.69    | 16.0min |
| 128  | 257.11 | 19.23 | 393.53    | 14.4min |
| **Best** | **Seed 67** | **Mean=307.48** | **Best Eval=633.69** |

**Algorithm: REINFORCE – b=mean**
| Seed | Mean   | Std    | Best Eval | Time    |
|------|--------|--------|-----------|---------|
| 20   | 247.30 |  17.97 | 292.68    | 34.8min |
| 42   | 408.19 | 111.54 | 1150.44   | 19.2min |
| 67   | 348.04 | 111.92 | 596.65    | 22.5min |
| 128  | 394.17 |  76.40 | 702.34    | 24.2min |
| **Best** | **Seed 42** | **Mean=408.19** | **Best Eval=1150.44** |

**Algorithm: Actor-Critic**
| Seed | Mean   | Std    | Best Eval | Time     |
|------|--------|--------|-----------|----------|
| 20   | 445.59 |  99.69 | 455.97    | 48.6min  |
| 42   | 658.34 | 125.68 | 590.08    | 76.8min  |
| 67   | 884.52 | 266.32 | 846.16    | 197.5min |
| 128  | 170.72 |  27.53 | 188.78    | 24.6min  |
| **Best** | **Seed 67** | **Mean=884.52** | **Best Eval=846.16** |

```
### Final Algorithm Comparison Summary – Hopper-v4
`comparison_summary.txt` actual results:
| Algorithm        | Best Seed | Conv. Ep | Final Avg | Final Std | Train Time |
|------------------|-----------|----------|-----------|-----------|------------|
| REINFORCE b=0    | 20        | 217      | 123.5     | 44.5      | 20.8min    |
| REINFORCE b=20   | 67        | 2323     | 307.5     | 17.8      | 16.0min    |
| REINFORCE b=mean | 42        | 426      | 408.2     | 111.5     | 19.2min    |
| Actor-Critic     | 67        | 223      | 884.5     | 266.3     | 3.3h       |

Notes: Conv. Ep   : First episode where 100-ep avg reward >= 200

---
## Hyperparameters

| Setting               | REINFORCE b=0   | REINFORCE b=20  | REINFORCE b=mean | Actor-Critic              |
|-----------------------|-----------------|-----------------|------------------|---------------------------|
| Episodes              | 10,000          | 10,000          | 10,000           | 10,000                    |
| Seeds                 | 20, 42, 67, 128 | 20, 42, 67, 128 | 20, 42, 67, 128  | 20, 42, 67, 128           |
| Discount factor (γ)   | 0.99            | 0.99            | 0.99             | 0.99                      |
| Learning rate         | 3e-4            | 3e-4            | 3e-4             | Actor: 3e-4 / Critic: 1e-3|
| Hidden units          | 64 (×2 layers)  | 64 (×2 layers)  | 64 (×2 layers)   | 64 (×2 layers)            |
| Activation            | Tanh            | Tanh            | Tanh             | Tanh                      |
| Weight init           | Orthogonal (√2) | Orthogonal (√2) | Orthogonal (√2)  | Orthogonal (√2)           |
| Initial σ             | 0.5             | 0.5             | 0.5              | 0.5                       |
| Loss reduction        | .mean()         | .mean()         | .mean()          | .mean()                   |
| Normalization         | None            | None            | mean + std       | TD advantage              |
| Grad clipping         | None            | None            | max_norm=1.0     | max_norm=1.0              |
| Entropy bonus (β)     | None            | None            | None             | 0.01, decays ×0.995/ep    |
| LR scheduler          | None            | None            | None             | StepLR(500, γ=0.7)        |

---
## Key Differences Between Algorithms
|                     | b=0          | b=20            | mean          | AC |
|---------------------|--------------|-----------------|---------------|---------------------------|
| Variance reduction  | None         | Partial         | Whitening     | Learned V(s)              |
| Bootstrap           | MC only      | MC only         | MC only       | TD(0)                     |
| Baseline type       | None         | Fixed scalar    | Episode mean  | Neural network            |
| Advantage scale     | Raw returns  | Shifted returns | Normalized ±1 | TD normalized             |
| Credit assignment   | Full episode | Full episode    | Full episode  | Per-step TD               |
| Exploration control | None         | None            | None          | Entropy bonus             |
| Stability mechanism | None         | None            | Grad clip 1.0 | Grad clip 0.1 + scheduler |

---
### Baseline Sweep (seed=67, 3,000 episodes)
Empirical sweep over constant baseline values `b ∈ {0, 5, 10, 20, 50, 100, 200}` to understand the effect of the baseline on learning performance. Run with `python sweep_baseline.py`.

| Baseline | Final Avg (100ep) | Converged |
|----------|-------------------|-----------|
| b=0      | 119.8             | Never     |
| b=5      | 178.2             | Never     |
| b=10     | 159.6             | Never     |
| b=20     | 161.5             | Never     |
| b=50     | 315.2             | Never     |
| b=100    | 340.3             | Never     |
| b=200    | 391.1             | Never     |
Full results saved to `Logs/sweep/sweep_summary.txt`.
---

## CSV Log Columns
Every training script writes a CSV with the following columns:
| Column                       | Description                               |
|------------------------------|-------------------------------------------|
| `episode`                    | Episode index                             |
| `reward`                     | Raw total reward for the episode          |
| `length`                     | Number of steps in the episode            |
| `avg_reward_100`             | Rolling 100-episode average reward        |
| `std_reward_100`             | Rolling 100-episode std of reward         |
| `policy_loss` / `actor_loss` | Policy gradient loss value                |
| `critic_loss`                | Critic MSE loss (Actor-Critic only)       |
| `entropy`                    | Mean action distribution entropy          |
| `episode_time_s`             | Wall-clock time for the episode (seconds) |
| `cumulative_time_s`          | Total elapsed training time (seconds)     |

---