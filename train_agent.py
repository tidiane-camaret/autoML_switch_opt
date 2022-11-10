import numpy as np


# test the agent
def eval_agent(env, policy, num_episodes=10, num_steps=100):
    actions, rewards = np.zeros((num_episodes, num_steps)), np.zeros((num_episodes, num_steps))
    for episode in range(num_episodes):
        obs = env.reset()
        for step in range(num_steps):
            action, _states = policy.predict(obs)
            obs, reward, done, info = env.step(action, )
            actions[episode, step] = action
            rewards[episode, step] = reward
            if done:
                break
    return actions, rewards
