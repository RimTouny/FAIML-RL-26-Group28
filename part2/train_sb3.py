import argparse
import os
import csv
import random
import time
import shutil
import numpy as np
import torch
import gymnasium as gym
import panda_gym  # type: ignore[import-not-found]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from rand_wrapper import RandomizationWrapper


# Hyperparameter search space
PPO_CONFIGS = {
    "ppo_cfg1": dict(
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
    ),
    "ppo_cfg2": dict(
        learning_rate=1e-4,
        n_steps=8192,
        batch_size=256,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.98,
        clip_range=0.1,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
    ),
    "ppo_cfg3": dict(
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.92,
        clip_range=0.3,
        ent_coef=0.01,
        vf_coef=0.4,
        max_grad_norm=1.0,
    ),
}

SAC_CONFIGS = {
    "sac_cfg1": dict(
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
    ),
    "sac_cfg2": dict(
        learning_rate=1e-3,
        buffer_size=500_000,
        learning_starts=5_000,
        batch_size=512,
        tau=0.01,
        gamma=0.98,
        train_freq=1,
        gradient_steps=2,
        ent_coef="auto",
        target_update_interval=1,
    ),
    "sac_cfg3": dict(
        learning_rate=7e-4,
        buffer_size=300_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=2,
        ent_coef="auto",
        target_update_interval=1,
    ),
}
# Reproducibility
def set_all_seeds(seed):
    """Seed Python random, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# CSV Logger Callback
class TrainingLoggerCallback(BaseCallback):
    """
    Logs per-episode reward, episode length, success rate, and wall-clock time.
    Requires the environment to be wrapped with Monitor.
    """

    def __init__(self, csv_path, verbose= 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.rewards = []
        self.successes = []
        self._ep_start = None
        self._train_start = None
        self._csv_file = None
        self._csv_writer = None
        # Track success over the whole episode. Panda-Gym may report
        # is_success before the terminal step, so relying only on the
        # final info dict can underestimate success rate.
        self._current_ep_success = False

    def _on_training_start(self):
        self._train_start = time.perf_counter()
        self._ep_start = time.perf_counter()
        # Open once — keeps handle alive for whole run (no per-episode open/close)
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["episode","reward","length","is_success","avg_reward_100", "std_reward_100", "success_rate_100", "episode_time_s","cumulative_time_s",])
        self._csv_file.flush()

    def _on_training_end(self):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if bool(info.get("is_success", False)):
                self._current_ep_success = True

            if "episode" not in info:
                continue

            reward = float(info["episode"]["r"])
            length = int(info["episode"]["l"])
            success = float(self._current_ep_success or bool(info.get("is_success", False)))
            self._current_ep_success = False

            now = time.perf_counter()
            ep_time = now - self._ep_start
            total_time = now - self._train_start
            self._ep_start = now

            self.rewards.append(reward)
            self.successes.append(success)

            reward_window = self.rewards[-100:]
            success_window = self.successes[-100:]

            avg_reward = float(np.mean(reward_window))
            std_reward = float(np.std(reward_window))
            success_rate = float(np.mean(success_window))
            episode_num = len(self.rewards)

            self._csv_writer.writerow([episode_num,round(reward, 4),length,int(success),round(avg_reward, 4),round(std_reward, 4),round(success_rate, 4),round(ep_time, 4),round(total_time, 2),])
            # Flush every 10 episodes — balances safety vs I/O overhead
            if episode_num % 10 == 0:
                self._csv_file.flush()

            if episode_num % 100 == 0:
                print(
                    f"  Episode {episode_num:5d} | "
                    f"Reward: {reward:8.3f} | "
                    f"Avg100: {avg_reward:8.3f} ± {std_reward:6.3f} | "
                    f"Success100: {success_rate:5.1%} | "
                    f"Time: {total_time / 60:.1f} min"
                )

        return True


# Environment and wrapper helpers
def make_panda_env(env_type,sampling_strategy = "none",udr_low = 0.5,udr_high = 6.0,):
    """
    Build PandaPush-v3 environment.

    - none: fixed source/target domain.
    - udr: sample object mass uniformly from [udr_low, udr_high].
    - adr: use adaptive domain randomization over [0.1, 10.0].
    """

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=env_type,
        reward_type="dense",
    )

    if sampling_strategy == "udr":
        env = RandomizationWrapper(env,mass_range=(udr_low, udr_high),mode="udr",)

    elif sampling_strategy == "adr":
        env = RandomizationWrapper(env,mass_range=(0.1, 10.0),mode="adr",adr_init_mass=1.0,adr_success_threshold=0.70, adr_expand_step=0.15, adr_shrink_step=0.05, adr_buffer_size=100, adr_center_lr=0.10,)
    return Monitor(env)


def get_randomization_wrapper(env):
    """Return the RandomizationWrapper inside a wrapper stack, if present."""
    current = env

    while True:
        if isinstance(current, RandomizationWrapper):
            return current

        if not hasattr(current, "env"):
            return None

        current = current.env


# Evaluation
def evaluate_with_success(model, env_type, n_eval_episodes = 50):
    """
    Final evaluation on a clean fixed source or target domain.
    No UDR/ADR wrapper is used here.
    """

    env = make_panda_env(env_type=env_type, sampling_strategy="none")

    returns = []
    successes = []

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_success = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_return += float(reward)
            done = terminated or truncated

            if bool(info.get("is_success", False)):
                ep_success = True

        returns.append(ep_return)
        successes.append(float(ep_success))

    env.close()

    return (float(np.mean(returns)),float(np.std(returns)),float(np.mean(successes)),)


# Plotting
def plot_reward_and_time(csv_path, save_path, title = "Training curve"):
    episodes, rewards, avgs, stds, times, success_rates = [], [], [], [], [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            avgs.append(float(row["avg_reward_100"]))
            stds.append(float(row["std_reward_100"]))
            times.append(float(row["episode_time_s"]))
            success_rates.append(float(row["success_rate_100"]))

    if len(episodes) < 2:
        print(f"  [WARN] Not enough episodes to plot: {len(episodes)}")
        return

    avgs = np.array(avgs)
    stds = np.array(stds)

    fig, axes = plt.subplots(3,1,figsize=(12, 10),gridspec_kw={"height_ratios": [3, 1, 1]},)

    ax1, ax2, ax3 = axes

    ax1.plot(episodes, rewards, alpha=0.5, label="Episode reward")
    ax1.plot(episodes, avgs, linewidth=2.0, label="100-episode avg")
    ax1.fill_between(episodes, avgs - stds, avgs + stds, alpha=0.2, label="±1 std")
    ax1.set_title(title)
    ax1.set_ylabel("Return")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.plot(episodes, times, label="Episode time")
    if len(times) >= 100:
        rolling_time = np.convolve(times, np.ones(100) / 100, mode="valid")
        ax2.plot(range(99, len(times)), rolling_time, label="100-episode avg")
    ax2.set_ylabel("Time (s)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    ax3.plot(episodes, success_rates, label="100-episode success rate")
    ax3.set_ylabel("Success")
    ax3.set_xlabel("Episode")
    ax3.set_ylim(0, 1)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved -> {save_path}")


# Save UDR / ADR evidence

def save_randomization_data(train_env, sampling_strategy, tag, udr_low, udr_high):
    """
    Save real mass samples and ADR history from the training wrapper.
    Returns the saved CSV path or None.
    """

    os.makedirs("results", exist_ok=True)
    rand_wrapper = get_randomization_wrapper(train_env)

    if rand_wrapper is None:
        return None

    if sampling_strategy == "udr" and getattr(rand_wrapper, "mass_samples", None):
        path = f"results/udr_mass_samples_{udr_low}_{udr_high}_{tag}.csv"

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "mass"])

            for idx, mass in enumerate(rand_wrapper.mass_samples, start=1):
                writer.writerow([idx, mass])

        print(f"  UDR mass samples saved -> {path}")
        return path

    if sampling_strategy == "adr" and getattr(rand_wrapper, "adr_history", None):
        path = f"results/adr_history_{tag}.csv"

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "center", "delta"])

            for episode, center, delta in rand_wrapper.adr_history:
                writer.writerow([episode, center, delta])

        print(f"  ADR history saved -> {path}")
        return path

    return None


# Train one run
def train_one(algo, cfg_name, cfg_kwargs, env_type, sampling_strategy, udr_low, udr_high, total_timesteps, seed, n_eval_episodes=50, device="cpu"):
    tag = f"{algo}_{cfg_name}_seed{seed}_{env_type}"

    if sampling_strategy == "udr":
        tag += f"_udr_{udr_low}_{udr_high}"
    elif sampling_strategy == "adr":
        tag += "_adr"

    log_dir = f"Logs/{tag}"
    model_dir = f"models/{tag}"
    best_model_dir = os.path.join(model_dir, "best")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    csv_path = f"{log_dir}/training_{tag}.csv"
    plot_path = f"{log_dir}/learning_curve_{tag}.png"

    final_model_path = f"{model_dir}/model_{tag}_final"
    best_model_zip = os.path.join(best_model_dir, "best_model.zip")

    print(f"\n{'-' * 70}")
    print(f"  algo={algo.upper()} | cfg={cfg_name} | seed={seed} | env={env_type}")
    print(f"  strategy={sampling_strategy} | timesteps={total_timesteps:,}")
    print(f"{'-' * 70}")

    # Seed everything first — before any env construction or model init
    set_all_seeds(seed)

    # Training environment: UDR/ADR applied only during training.
    train_env = make_panda_env(env_type=env_type,sampling_strategy=sampling_strategy,udr_low=udr_low,udr_high=udr_high,)
    train_env.reset(seed=seed)
    train_env.action_space.seed(seed)
    train_env.observation_space.seed(seed)

    # Evaluation callback environment: clean fixed domain.
    callback_eval_env = make_panda_env(env_type=env_type,sampling_strategy="none",)
    callback_eval_env.reset(seed=seed + 1000)

    shared_kwargs = dict(policy="MultiInputPolicy",verbose=1,seed=seed,**cfg_kwargs,)

    # Canonical SB3 positional constructor: Algo(policy, env, ...)
    model_kwargs = {k: v for k, v in shared_kwargs.items() if k != "policy"}
    if algo == "ppo":
        model = PPO("MultiInputPolicy", train_env, device=device, **model_kwargs)
    elif algo == "sac":
        model = SAC("MultiInputPolicy", train_env, device=device, **model_kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    logger_callback = TrainingLoggerCallback(csv_path=csv_path)

    eval_callback = EvalCallback(callback_eval_env,best_model_save_path=best_model_dir,log_path=os.path.join(log_dir, "eval"),eval_freq=10_000,n_eval_episodes=10,deterministic=True,render=False,verbose=1,)

    callbacks = CallbackList([logger_callback, eval_callback])

    model.learn(total_timesteps=total_timesteps,callback=callbacks,reset_num_timesteps=True,progress_bar=True,)

    model.save(final_model_path)
    print(f"  Final model saved -> {final_model_path}.zip")

    randomization_csv = save_randomization_data(train_env=train_env,sampling_strategy=sampling_strategy,tag=tag,udr_low=udr_low,udr_high=udr_high,)

    # Evaluate the best callback checkpoint if it exists; otherwise use final model.
    if os.path.exists(best_model_zip):
        print(f"  Best callback model found -> {best_model_zip}")

        if algo == "ppo":
            eval_model = PPO.load(best_model_zip, device=device)
        else:
            eval_model = SAC.load(best_model_zip, device=device)

        selected_model_path = best_model_zip.replace(".zip", "")
        selected_model_type = "callback_best"
    else:
        print("  No callback best model found. Using final model.")
        eval_model = model
        selected_model_path = final_model_path
        selected_model_type = "final"

    mean_return, std_return, success_rate = evaluate_with_success(eval_model,env_type=env_type,n_eval_episodes=n_eval_episodes,)

    # Cross-domain evaluation: train on source → test on target gives the
    # sim-to-real lower bound; train on target → test on source is a sanity check.
    cross_domain = "target" if env_type == "source" else "source"
    mean_return_cross, std_return_cross, success_rate_cross = evaluate_with_success(eval_model,env_type=cross_domain,n_eval_episodes=n_eval_episodes,)

    print(f"\n  Evaluation of selected model ({n_eval_episodes} episodes)")
    print(f"  Selected model type      : {selected_model_type}")
    print(f"  Mean return ({env_type:>6})  : {mean_return:.3f} +/- {std_return:.3f}")
    print(f"  Success rate ({env_type:>6}) : {success_rate:.1%}")
    print(f"  Mean return ({cross_domain:>6})  : {mean_return_cross:.3f} +/- {std_return_cross:.3f}")
    print(f"  Success rate ({cross_domain:>6}) : {success_rate_cross:.1%}  [cross-domain / sim-to-real]")
    print(f"  Selected path            : {selected_model_path}")

    plot_reward_and_time(
        csv_path,
        plot_path,
        title=f"{algo.upper()} {cfg_name} seed={seed} - {env_type} [{sampling_strategy}]",
    )

    train_env.close()
    callback_eval_env.close()

    return (mean_return,std_return,success_rate,selected_model_path,csv_path,plot_path,selected_model_type,randomization_csv,)


# Main

def main():
    parser = argparse.ArgumentParser(description="Train PPO or SAC on PandaPush-v3")

    parser.add_argument("--algo", default="sac", choices=["ppo", "sac"])
    parser.add_argument("--env-type", default="source", choices=["source", "target"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", nargs="+", type=int, default=[42, 67, 128])
    parser.add_argument("--sampling-strategy", default="none", choices=["none", "udr", "adr"])
    parser.add_argument("--udr-low", type=float, default=0.5)
    parser.add_argument("--udr-high", type=float, default=6.0)
    parser.add_argument("--n-eval-episodes", type=int, default=50)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Algorithm         : {args.algo.upper()}")
    print(f"Environment type  : {args.env_type}")
    print(f"Sampling strategy : {args.sampling_strategy}")
    print(f"Timesteps         : {args.timesteps:,}")
    print(f"Seeds             : {args.seed}")
    print(f"Device            : {device}")
    if args.sampling_strategy == "udr":
        print(f"UDR range         : [{args.udr_low}, {args.udr_high}]")
    print("=" * 70)

    configs = PPO_CONFIGS if args.algo == "ppo" else SAC_CONFIGS

    # Store both success rate and return so ranking is meaningful when success ties.
    results = {
        cfg_name: {
            "success": [],
            "return": [],
        }
        for cfg_name in configs
    }

    all_runs = {}

    for cfg_name, cfg_kwargs in configs.items():
        for seed in args.seed:
            result = train_one(
                algo=args.algo,
                cfg_name=cfg_name,
                cfg_kwargs=cfg_kwargs,
                env_type=args.env_type,
                sampling_strategy=args.sampling_strategy,
                udr_low=args.udr_low,
                udr_high=args.udr_high,
                total_timesteps=args.timesteps,
                seed=seed,
                n_eval_episodes=args.n_eval_episodes,
                device=device,
            )

            mean_return, std_return, success_rate, selected_path, csv_path, plot_path, model_type, randomization_csv = result

            results[cfg_name]["success"].append(success_rate)
            results[cfg_name]["return"].append(mean_return)

            all_runs[(cfg_name, seed)] = result

    # Rank configs by:
    # 1. average success rate
    # 2. average return as tie-breaker
    cfg_success = {
        cfg_name: float(np.mean(results[cfg_name]["success"]))
        for cfg_name in configs
    }

    cfg_return = {
        cfg_name: float(np.mean(results[cfg_name]["return"]))
        for cfg_name in configs
    }

    best_cfg = max(
        configs,
        key=lambda cfg_name: (
            cfg_success[cfg_name],
            cfg_return[cfg_name],
        ),
    )

    # Rank seeds inside the best config by:
    # 1. success rate
    # 2. mean return as tie-breaker
    best_seed = max(
        args.seed,
        key=lambda seed: (
            all_runs[(best_cfg, seed)][2],
            all_runs[(best_cfg, seed)][0],
        ),
    )

    best_run = all_runs[(best_cfg, best_seed)]

    best_tag = f"{args.algo}_push_{args.sampling_strategy}_{args.env_type}_best"

    best_model_dir = f"models/{best_tag}"
    best_log_dir = f"Logs/{best_tag}"

    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(best_log_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    src_zip = best_run[3] + ".zip"
    dst_zip = f"{best_model_dir}/model_{best_tag}.zip"

    if os.path.exists(src_zip):
        shutil.copy2(src_zip, dst_zip)
    else:
        print(f"Warning: selected model file not found: {src_zip}")

    dst_csv = f"{best_log_dir}/training_{best_tag}.csv"
    dst_plot = f"{best_log_dir}/learning_curve_{best_tag}.png"

    shutil.copy2(best_run[4], dst_csv)
    shutil.copy2(best_run[5], dst_plot)

    # Copy selected UDR/ADR evidence to canonical filenames for analysis_plots.py.
    selected_randomization_csv = best_run[7]
    canonical_randomization_csv = "N/A"

    if selected_randomization_csv and os.path.exists(selected_randomization_csv):
        if args.sampling_strategy == "udr":
            canonical_randomization_csv = f"results/udr_mass_samples_{args.udr_low}_{args.udr_high}.csv"
            shutil.copy2(selected_randomization_csv, canonical_randomization_csv)

        elif args.sampling_strategy == "adr":
            canonical_randomization_csv = "results/adr_history.csv"
            shutil.copy2(selected_randomization_csv, canonical_randomization_csv)

    sep = "=" * 100

    lines = [
        sep,
        f"HYPERPARAMETER SEARCH SUMMARY - {args.algo.upper()}",
        "Ranked by: average success rate first, then average return",
        f"Environment type : {args.env_type}",
        f"Strategy         : {args.sampling_strategy}",
        f"Timesteps/run    : {args.timesteps:,}",
        f"Evaluation eps   : {args.n_eval_episodes}",
        sep,
        f"{'Config':<12} {'Avg success':>12} {'Avg return':>12}  Per-seed results",
        f"{'-' * 12:<12} {'-' * 12:>12} {'-' * 12:>12}  {'-' * 50}",
    ]

    for cfg_name in configs:
        per_seed = "  ".join([
            (
                f"seed{seed}: "
                f"suc={all_runs[(cfg_name, seed)][2]:.1%}, "
                f"ret={all_runs[(cfg_name, seed)][0]:.2f}"
            )
            for seed in args.seed
        ])

        marker = "  <-- BEST" if cfg_name == best_cfg else ""

        lines.append(
            f"{cfg_name:<12} "
            f"{cfg_success[cfg_name]:>12.1%} "
            f"{cfg_return[cfg_name]:>12.3f}  "
            f"{per_seed}{marker}"
        )

    lines += [
        "",
        f"Best config       : {best_cfg}",
        f"Best seed         : {best_seed}",
        f"Best model type   : {best_run[6]}",
        f"Best return       : {best_run[0]:.3f} +/- {best_run[1]:.3f}",
        f"Best success rate : {best_run[2]:.1%}",
        "",
        f"Best model source : {src_zip}",
        f"Best model copy   : {dst_zip}",
        f"Best CSV copy     : {dst_csv}",
        f"Best plot copy    : {dst_plot}",
        f"Randomization CSV : {canonical_randomization_csv}",
        sep,
        "",
        f"Winning hyperparameters ({best_cfg}):",
    ]

    for key, value in configs[best_cfg].items():
        lines.append(f"  {key:<28} = {value}")

    lines.append("")

    print("\n" + "\n".join(lines))

    summary_path = f"results/summary_{best_tag}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()