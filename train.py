from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from env import env_name
import gymnasium as gym
import ale_py
import gc

gym.register_envs(ale_py)

# Create and wrap the Atari environment
env = make_atari_env(env_name, n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)  # Stack frames for the DQN model

policy_kwargs = dict(features_extractor_kwargs=dict(features_dim=512))

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    buffer_size=100000,
    learning_starts=10000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.1,
    train_freq=4,
    policy_kwargs=policy_kwargs,
    verbose=1,
)

class GCCallback(CheckpointCallback):
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            gc.collect()
        return super()._on_step()

checkpoint_callback = GCCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="dqn"
)

total_timesteps = 1_000_000
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=True)

# Save the model
model.save("dqn_final")

env.close()
