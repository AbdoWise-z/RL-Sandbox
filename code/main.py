import gym
import numpy as np
import matplotlib.pyplot as plt
from reinforce_torch import Agent
from gym.wrappers.record_video import RecordVideo
from lunar_lime import explain_action
from PIL import Image
import os

def plotLearning(scores, filename, window=100):
    running_avg = np.zeros(len(scores))
    for i in range(len(scores)):
        running_avg[i] = np.mean(scores[max(0, i-window):(i+1)])
    plt.plot(running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    env = gym.make('LunarLander-v2', render_mode="rgb_array") 
    video_path = "./videos/"
    env = RecordVideo(env, video_path, episode_trigger=lambda x: x % 100 == 0)  # Record every 100 episodes

    agent = Agent(alpha=0.0005, gamma=0.99, n_actions=4, input_dims=env.observation_space.shape[0])
    score_history = []
    num_episodes = 1001

    for i in range(num_episodes):
        done = False
        score = 0
        observation, info = env.reset()
        episode_explanations = []  # Store explanations for multiple steps
        episode_images = []  # Store image filenames
        
        while not done:
            action, action_probs = agent.choose_action(observation, return_probs=True)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(observation, action, reward)
            score += reward

            # Explain some actions during the episode
            if i % 100 == 0 and np.random.rand() < 0.2:  # Explain ~20% of the steps
                print(f"Explaining action at step {len(episode_explanations) + 1}")
                
                # Capture Screenshot (Fixing the render issue)
                try:
                    frame = env.render()
                    if frame is None:
                        raise ValueError("Render did not return a frame")
                    os.makedirs(f"./screenshots/episode_{i}", exist_ok=True)
                    image_path = f"./screenshots/episode_{i}/screenshot_{i}_{len(episode_images)}.png"
                    Image.fromarray(frame).save(image_path)  # Save image
                    episode_images.append(image_path)
                except Exception as e:
                    print(f"Could not capture screenshot: {e}")
                    image_path = None

                # Generate Explanation
                explanation_html = explain_action(agent, observation, return_html=True)
                
                # Embed image in explanation if available
                if image_path:
                    # Add image to explanation and center it
                    explanation_html += f'<br><img src="../{image_path}" width="500" style="display: block; margin-left: auto; margin-right: auto;"><br><hr>'
                episode_explanations.append(explanation_html)

            observation = observation_

        score_history.append(score)
        agent.learn()
        avg_score = np.mean(score_history[-100:])
        print(f'episode: {i}, score: {score:.1f}, average score: {avg_score:.1f}')

        # Save all explanations & images in one HTML file
        if i % 100 == 0 and episode_explanations:
            os.makedirs(f"./explanations", exist_ok=True)
            with open(f"./explanations/lime_explanation_episode_{i}.html", "w", encoding="utf-8") as f:
                for explanation in episode_explanations:
                    f.write(explanation)

    agent.save_model("lunar_lander_reinforce.pth")
    filename = 'lunar-lander.png'
    plotLearning(score_history, filename=filename, window=100)