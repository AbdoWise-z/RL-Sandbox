import gym
from reinforce_torch import Agent


def try_one_round():
    # loads and runs the model for one episode in human mode to see how it performs
    env = gym.make('LunarLander-v2', render_mode="human")
    agent = Agent(alpha=0.0005, gamma=0.99, n_actions=4, input_dims=env.observation_space.shape[0])
    agent.load_model("lunar_lander_reinforce_2000.pth")
    score = 0
    observation, info = env.reset()
    done = False
    while not done:
        action, action_probs = agent.choose_action(observation, return_probs=True)
        observation_, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        agent.store_transition(observation, action, reward)
        score += reward
        observation = observation_
        env.render()
    print(f"Score: {score}")
    env.close()


if __name__ == "__main__":
    try_one_round()