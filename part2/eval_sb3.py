from __future__ import annotations

import argparse
import csv
import os

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor


def _get_sim(env):
    """Walk the wrapper stack to the base panda_gym env and return its sim."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base.task.sim


def _set_mass(env, mass):
    """Inject a custom cube mass into the running PyBullet simulation."""
    sim = _get_sim(env)
    body_id = sim._bodies_idx["object"]
    sim.physics_client.changeDynamics(bodyUniqueId=body_id,linkIndex=-1,mass=float(mass),)


def _load_model(model_path):
    """Load PPO or SAC based on the model filename."""
    fname = os.path.basename(model_path).lower()

    if "ppo" in fname:
        return PPO.load(model_path)

    if "sac" in fname:
        return SAC.load(model_path)

    raise ValueError(
        "Could not infer algorithm from model filename. "
        "The filename should contain 'ppo' or 'sac'."
    )


def evaluate(model_path,n_episodes,deterministic,render,env_type,custom_mass = None,):
    """
    Returns (mean_return, std_return, success_rate).

    When custom_mass is passed, the object mass is injected before reset
    in each episode.
    """

    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure the path points to a saved SB3 model."
        )

    render_mode = "human" if render else "rgb_array"

    env = gym.make("PandaPush-v3",render_mode=render_mode,type=env_type,reward_type="dense",)
    env = Monitor(env)

    model = _load_model(model_path)

    episode_returns = []
    successes = []

    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset()

        if custom_mass is not None:
            _set_mass(env, custom_mass)


        terminated = False
        truncated = False
        episode_return = 0.0
        ep_success = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += float(reward)

            if bool(info.get("is_success", False)):
                ep_success = True

        episode_returns.append(episode_return)
        successes.append(float(ep_success))

        print(
            f"Episode {episode:03d} | "
            f"return = {episode_return:.3f} | "
            f"success = {int(ep_success)}"
        )

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    success_rate = float(np.mean(successes)) if successes else 0.0

    mass_tag = f"{custom_mass:.1f} kg" if custom_mass is not None else env_type

    print(f"\n{'=' * 52}")
    print(f"  Model        : {os.path.basename(model_path)}")
    print(f"  Domain/Mass  : {mass_tag}")
    print(f"{'=' * 52}")
    print(f"  Episodes     : {n_episodes}")
    print(f"  Mean return  : {returns.mean():.3f}")
    print(f"  Std  return  : {returns.std():.3f}")
    print(f"  Min  return  : {returns.min():.3f}")
    print(f"  Max  return  : {returns.max():.3f}")
    print(f"  Success rate : {success_rate:.1%}")
    print(f"{'=' * 52}\n")

    return float(returns.mean()), float(returns.std()), success_rate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on PandaPush-v3")
    parser.add_argument("--model-path",type=str,required=True,help="Path to a saved PPO/SAC model zip file.",)
    parser.add_argument("--episodes",type=int,default=50,help="Number of evaluation episodes. Default is 50 for the project report.",)
    parser.add_argument("--stochastic",action="store_true",help="Use stochastic policy sampling instead of deterministic actions.",)
    parser.add_argument("--render",action="store_true",help="Render with a visible window.",)
    parser.add_argument("--env-type",type=str,default="target",choices=["source", "target"],help="Environment type to evaluate on.",)
    parser.add_argument("--mass",type=float,default=None,help="Optional custom cube mass in kg.",)
    parser.add_argument("--sensitivity",action="store_true",help="Sweep masses [1, 2, 3, 5, 8, 10] kg.",)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sensitivity:
        masses = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]

        base_name = os.path.basename(args.model_path).replace(".zip", "")
        os.makedirs("results", exist_ok=True)
        save_path = f"results/sensitivity_{base_name}.csv"

        print(f"\n{'=' * 52}")
        print("  SENSITIVITY ANALYSIS")
        print(f"  Model    : {os.path.basename(args.model_path)}")
        print(f"  Env type : {args.env_type}")
        print(f"{'=' * 52}")
        print(f"  {'Mass (kg)':<12} {'Mean return':>12} {'Std':>8} {'Success':>10}")
        print(f"  {'-' * 9:<12} {'-' * 11:>12} {'-' * 3:>8} {'-' * 7:>10}")

        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mass", "mean_return", "std_return", "success_rate", "episodes"])

            for mass in masses:
                mean_r, std_r, suc_r = evaluate(model_path=args.model_path,n_episodes=args.episodes,deterministic=not args.stochastic,render=False,env_type=args.env_type,custom_mass=mass,)

                print(f"  {mass:<12.1f} {mean_r:>12.3f} {std_r:>8.3f} {suc_r:>9.1%}")
                writer.writerow([mass, mean_r, std_r, suc_r, args.episodes])

        print(f"\n  Saved -> {save_path}")
        print(f"{'=' * 52}\n")

    else:
        evaluate(model_path=args.model_path,n_episodes=args.episodes,deterministic=not args.stochastic,render=args.render,env_type=args.env_type,custom_mass=args.mass,)