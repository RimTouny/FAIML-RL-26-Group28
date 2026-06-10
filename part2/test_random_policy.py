"""Test a random policy on the Gym Hopper environment

    Play around with this code to get familiar with the
    Hopper environment.

    For example, what happens if you don't reset the environment
    even after the episode is over?
    When exactly is the episode over?
    What is an action here?
"""

import gymnasium as gym
import numpy as np
import random
import torch
import panda_gym  
import argparse

# Reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-type", type=str, default="source",choices=["source", "target"],help="source = 1 kg cube,  target = 5 kg cube")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    env = gym.make("PandaPush-v3",render_mode="human" if args.render else "rgb_array",type=args.env_type,reward_type="dense")

    print("\n===================================")
    print(f"PANDA PUSH  [{args.env_type.upper()}  domain]")
    print("===================================\n")

    # STATE SPACE
    print("1. OBSERVATION SPACE")
    print("--------------------------")
    print(env.observation_space)   # state-space

    state, info = env.reset(seed=42)

    print("\nKeys in state dictionary:")
    print(state.keys())

    print("\nObservation Shape:")
    print(state["observation"].shape)

    print("\nAchieved Goal Shape:")
    print(state["achieved_goal"].shape)

    print("\nDesired Goal Shape:")
    print(state["desired_goal"].shape)

    # ACTION SPACE
    print("\n\n2. ACTION SPACE")
    print("--------------------------")
    print(env.action_space) # action-space

    print("\nShape:")
    print(env.action_space.shape)

    print("\nLow Bound:")
    print(env.action_space.low)

    print("\nHigh Bound:")
    print(env.action_space.high)

    # INITIAL STATE
    print("\n\n3. INITIAL STATE")
    print("--------------------------")

    print("\nObservation:")
    print(state["observation"])

    print("\nAchieved Goal:")
    print(state["achieved_goal"])

    print("\nDesired Goal:")
    print(state["desired_goal"])

    # RANDOM POLICY WITH RESET
    n_episodes = 10

    print("\n\n===================================")
    print("RANDOM POLICY WITH RESET")
    print("===================================")

    for ep in range(n_episodes):

        state, info = env.reset() # Reset environment to initial state
        done = False
        total_reward = 0
        steps = 0

        while not done:  # Until the episode is over
            action = env.action_space.sample() # Sample random action
            next_state, reward, terminated, truncated, info = env.step(action) # Step the simulator to the next timestep
            done = terminated or truncated
            total_reward += reward
            steps += 1
            state = next_state

            if args.render:
                env.render()

        print(f"Episode {ep+1}")
        print(f"Steps: {steps}")
        print(f"Total Reward: {total_reward:.2f}")
        print("--------------------------")

    # RANDOM POLICY WITHOUT RESET
    print("\n\n===================================")
    print("RANDOM POLICY WITHOUT RESET")
    print("===================================")

    state, info = env.reset()

    for ep in range(n_episodes):

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            state = next_state

            if args.render:
                env.render()

        print(f"Episode {ep+1}")
        print(f"Steps: {steps}")
        print(f"Total Reward: {total_reward:.2f}")
        print("--------------------------")

    env.close()


if __name__ == "__main__":
    main()