from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from env import env_name
import gymnasium as gym
import ale_py
import gc
import torch

# Register Atari environments
gym.register_envs(ale_py)


# Custom CNN feature extractor
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        # Extract number of input channels from observation space
        n_input_channels = observation_space.shape[0]

        # Define CNN layers
        cnn = nn.Sequential(
            # First layer: CNN with filter size 16 and stride 4
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Second layer: CNN with filter size 8 and stride 2
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the size of flattened features
        with torch.no_grad():
            n_flatten = cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # Initialize parent class with calculated features_dim
        super().__init__(observation_space, features_dim=n_flatten)

        self.cnn = cnn

    def forward(self, observations):
        return self.cnn(observations)


# Create and wrap the Atari environment
env = make_atari_env(env_name, n_envs=4, seed=0)  # Using multiple environments for PPO
env = VecFrameStack(env, n_stack=4)

# Configure the policy with our custom feature extractor
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
)

# Create the PPO model
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=2.5e-4,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    clip_range_vf=None,
    normalize_advantage=True,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    verbose=1,
)


# Garbage collection callback
class GCCallback(CheckpointCallback):
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            gc.collect()
        return super()._on_step()


# Set up checkpoint callback
checkpoint_callback = GCCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="ppo"
)

# Train the model
total_timesteps = 1_000_000
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=True)

# Save the final model
model.save("ppo_final")
env.close()