import shap
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import gymnasium as gym
import ale_py
import time
from env import env_name
import warnings
import matplotlib.pyplot as plt

# Suppress the warning
warnings.filterwarnings("ignore", category=UserWarning, message="unrecognized nn.Module: Flatten")

gym.register_envs(ale_py)

def test_model_with_shap(model_path, num_episodes=10, render=True):
    # Create and wrap the environment
    env = make_atari_env(env_name, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)

    # Load the trained model
    model = DQN.load(model_path)

    # Extract the PyTorch model from the DQN model
    pytorch_model = model.policy.q_net

    # Check if the model is on GPU and move it to CPU for SHAP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model = pytorch_model.to(device)

    # Create a SHAP DeepExplainer
    background = torch.zeros((1, 4, 84, 84)).to(device)  # Background data for SHAP (must be a PyTorch tensor)
    explainer = shap.DeepExplainer(pytorch_model, background)

    episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Convert observation to PyTorch tensor and reshape for SHAP
            obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # Shape: (1, 4, 84, 84)

            # Get model's action
            action, _ = model.predict(obs, deterministic=True)

            # Generate SHAP values for the current observation
            shap_values = explainer.shap_values(obs_tensor)

            # Take step in environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]  # Reward comes as array due to vectorized env

            if render:
                # Render the environment
                env.render("human")

                # Display SHAP values for the current observation
                plt.figure(figsize=(10, 5))
                print(np.shape(shap_values))
                sv_agg  = np.mean(shap_values, axis=1)
                obs_agg = np.mean(obs_tensor.cpu().numpy(), axis=1)
                shap.image_plot(sv_agg, -obs_agg, show=False)
                plt.title(f"SHAP Values at Time Step (Episode {episode + 1})")
                plt.show()

                time.sleep(0.025)  # Add small delay to make rendering viewable

            if done:
                print(f"Episode {episode + 1} reward: {episode_reward}")
                episode_rewards.append(episode_reward)
                break

    env.close()
    # Print summary statistics
    print("\nTest Results:")
    print(f"Number of episodes: {num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Standard deviation: {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")

    return episode_rewards, shap_values


if __name__ == "__main__":
    model_path = "dqn_final.zip"

    # Test the model with SHAP explainability
    rewards, shap_values = test_model_with_shap(
        model_path=model_path,
        num_episodes=10,
        render=True
    )