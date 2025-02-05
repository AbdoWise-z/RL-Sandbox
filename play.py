from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import numpy as np
import gymnasium as gym
import ale_py
import time
from env import env_name
from lime import lime_image
from skimage.segmentation import quickshift, mark_boundaries
import matplotlib.pyplot as plt
import torch

gym.register_envs(ale_py)


def custom_segmentation(image):
    """Handle grayscale Atari frames for LIME segmentation"""
    # Extract last frame from the stack
    last_frame = image[:, :, -1]

    # Convert to pseudo-RGB by repeating grayscale channel
    pseudo_rgb = np.repeat(last_frame[:, :, np.newaxis], 3, axis=-1)

    # Normalize if needed
    if pseudo_rgb.max() > 1.0:
        pseudo_rgb = pseudo_rgb.astype(np.float32) / 255.0

    # Perform segmentation with adapted parameters
    return quickshift(
        pseudo_rgb,
        kernel_size=3,
        max_dist=5,
        ratio=0.2,
        convert2lab=False  # Disable Lab conversion
    )


def test_model(model_path, num_episodes=10, render=True, explain_steps=False):
    # Create environment
    env = make_atari_env(env_name, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)

    # Load model
    model = DQN.load(model_path)

    episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            new_obs, reward, done, info = env.step(action)
            episode_reward += reward[0]

            # Generate explanations at specified intervals
            if explain_steps and step_count % 10 == 3:
                current_obs = obs[0]  # (84, 84, 4) array

                # LIME classifier wrapper
                def classifier_fn(images):
                    """Convert images to model input format"""
                    tensor = torch.from_numpy(images).float().to(model.device)
                    tensor = tensor.permute(0, 3, 1, 2)  # NHWC to NCHW
                    with torch.no_grad():
                        q_values = model.policy.q_net(tensor)
                    return q_values.cpu().numpy()

                # Create explainer and generate explanation
                explainer = lime_image.LimeImageExplainer()
                explanation = explainer.explain_instance(
                    current_obs,
                    classifier_fn,
                    labels=(action[0],),
                    top_labels=3,
                    hide_color=0,
                    num_samples=500,
                    segmentation_fn=custom_segmentation
                )

                # Visualize results
                temp, mask = explanation.get_image_and_mask(
                    action[0],
                    positive_only=True,
                    num_features=5,
                    hide_rest=False
                )

                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.imshow(current_obs[:, :, -1], cmap='gray')
                plt.title("Original Frame")

                plt.subplot(122)
                plt.imshow(mark_boundaries(temp[:, :, 0], mask), cmap='gray')
                plt.title("LIME Explanation")
                plt.show()

            if render:
                env.render("human")
                time.sleep(0.025)

            obs = new_obs
            step_count += 1

            if done:
                print(f"Episode {episode + 1} reward: {episode_reward}")
                episode_rewards.append(episode_reward)
                break

    env.close()

    # Print summary
    print("\nTest Results:")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Std deviation: {np.std(episode_rewards):.2f}")
    return episode_rewards


if __name__ == "__main__":
    model_path = "dqn_final.zip"
    rewards = test_model(
        model_path=model_path,
        num_episodes=1,  # Start with fewer episodes for testing
        render=True,
        explain_steps=True
    )