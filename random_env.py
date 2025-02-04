import gymnasium as gym
import ale_py

# Register environment
gym.register_envs(ale_py)

# Create and wrap the Atari environment
env = gym.make("ALE/DemonAttack-v5", render_mode="human")
env.reset()

# Render the environment
for _ in range(1000):  # Run for a few steps to see the rendering
    env.render()

    action = env.action_space.sample()  # Random action
    env.step(action)

env.close()
