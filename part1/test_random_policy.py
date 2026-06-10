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

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def main():

    render = False

    # Create environment
    if render:
        env = gym.make('Hopper-v4', render_mode='human')
    else:
        env = gym.make('Hopper-v4', render_mode='rgb_array')

    # TASK 1 - ENVIRONMENT EXPLORATION
    print("\n==============================")
    print("HOPPER ENVIRONMENT INFORMATION")
    print("==============================\n")

    # State Space
    print("1. STATE SPACE")
    print("-------------------")
    print("Observation Space:")
    print(env.observation_space)

    print("\nState Space Shape:")
    print(env.observation_space.shape)

    print("\nLow Values:")
    print(env.observation_space.low)

    print("\nHigh Values:")
    print(env.observation_space.high)

    print(f"State Space: {env.observation_space}")
    print(f"Dimension: {env.observation_space.shape[0]}")
    print("Type: Continuous (Box space)")


    # Action Space
    print("\n\n2. ACTION SPACE")
    print("-------------------")
    print("Action Space:")
    print(env.action_space)

    print("\nAction Space Shape:")
    print(env.action_space.shape)

    print("\nAction Low:")
    print(env.action_space.low)

    print("\nAction High:")
    print(env.action_space.high)

    print("\nAction Space Type:")
    print("Continuous")

    # Reset environment
    state, info = env.reset(seed=42)

    print("\n\n3. INITIAL STATE")
    print("-------------------")
    print(state)

    print("\nState Dimension:")
    print(len(state))

    # MuJoCo model information
    model = env.unwrapped.model
  
    print("\n\n4. BODY NAMES")
    print("-------------------")

    body_names = []

    for i in range(model.nbody):

        body_name = model.body(i).name
        body_names.append(body_name)

        print(f"Body {i}: {body_name}")

    print("\n\n5. BODY MASSES (kg)")
    print("--------------------------------------")


    for i in range(model.nbody):

        body_name = model.body(i).name
        body_mass = model.body_mass[i]
        print(f"{body_name}: {body_mass}")

    # source_masses = model.body_mass.copy()
    # # Example target domain:
    # # increase all masses by 1.5x
    # target_masses = source_masses * 1.5

    # print(f"{'Body':15s} {'Source Mass':15s} {'Target Mass':15s}")

    # for i in range(model.nbody):

    #     body_name = model.body(i).name

    #     source_mass = source_masses[i]
    #     # target_mass = target_masses[i]

    #     # print(f"{body_name:15s} {source_mass:<15.4f} {target_mass:<15.4f}")


    print("\n\n6. DEGREES OF FREEDOM (DoFs)")
    print("-------------------")
    print("Number of DoFs:", model.nv)

    print("\nDoFs for each body:")
    for i in range(model.nbody):
        body_name = model.body(i).name
        dof = model.body_dofnum[i]
        print(f"{body_name}: {dof}")

    print("\n\n7. ACTUATORS")
    print("-------------------")
    print("Number of actuators:", model.nu)

    # RANDOM POLICY TEST
    n_episodes = 50

    print("\n\n============================================================")
    print("RUNNING RANDOM POLICY- Restarting the Environment")
    print("============================================================")
    
    for ep in range(n_episodes):

        done = False
        state, info = env.reset()   # Reset environment to initial state

        total_reward = 0
        step = 0

        while not done:

            # Random action
            action = env.action_space.sample() # Sample random action

            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)  # Step the simulator to the next timestep

            done = terminated or truncated

            total_reward += reward
            step += 1

            state = next_state

            if render:
                env.render()

        print(f"Episode {ep+1}")
        print(f"Steps: {step}")
        print(f"Total Reward: {total_reward:.2f}")
        print("-------------------")


    print("\n\n============================================================")
    print("RUNNING RANDOM POLICY- Without Restarting the Environment")
    print("============================================================")

    # Reset once before continuous execution
    state, info = env.reset()

    for ep in range(n_episodes):

        done = False
        # state, info = env.reset()  # Reset environment to initial state

        total_reward = 0
        step = 0

        while not done:  # Until the episode is over

            # Random action
            action = env.action_space.sample()  # Sample random action

            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)  # Step the simulator to the next timestep

            done = terminated or truncated

            total_reward += reward
            step += 1

            state = next_state

            if render:
                env.render()

        print(f"Episode {ep+1}")
        print(f"Steps: {step}")
        print(f"Total Reward: {total_reward:.2f}")
        print("-------------------")

    env.close()


if __name__ == '__main__':
    main()    