import gym
import numpy as np
import pandas as pd
from reinforce_torch import Agent

def try_one_round(env, agent):
    trajectory = []  # Store (observation, action, reward) tuples
    score = 0
    observation, info = env.reset()
    done = False

    while not done:
        action, action_probs = agent.choose_action(observation, return_probs=True)
        observation_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        trajectory.append((observation, action, reward))
        score += reward
        observation = observation_

    print(f"Score: {score}")

    # Compute Q-values using reward-to-go approach
    observations = []
    actions = []
    q_values = []
    G = 0  # Initialize return (Q-value)

    for obs, action, reward in reversed(trajectory):
        G = reward + agent.gamma * G  # Compute discounted return
        observations.append(obs)
        actions.append(action)
        q_values.append(G)

    # Reverse lists to match time order
    observations.reverse()
    actions.reverse()
    q_values.reverse()

    return observations, actions, q_values

if __name__ == "__main__":
    num_episodes = 10
    env = gym.make('LunarLander-v2')  # No need for render mode
    agent = Agent(alpha=0.0005, gamma=0.99, n_actions=4, input_dims=env.observation_space.shape[0])
    agent.load_model("lunar_lander_reinforce_2000.pth")

    all_observations = []
    all_actions = []
    all_q_values = []

    for episode in range(num_episodes):
        obs, acts, q_vals = try_one_round(env, agent)
        all_observations.extend(obs)  # Append new values
        all_actions.extend(acts)
        all_q_values.extend(q_vals)

    env.close()

    # Save to CSV
    data = np.column_stack([np.array(all_observations), np.array(all_actions), np.array(all_q_values)])
    obs_dim = len(all_observations[0])  # Get observation space size
    columns = [f"obs_{i}" for i in range(obs_dim)] + ["action", "q_value"]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv("lunar_lander_data.csv", index=False)
    print("Saved data to lunar_lander_data.csv")