import numpy as np



def eval_agent(env, policy, num_episodes=1, num_steps=5):
    actions, rewards, infos = np.zeros((num_episodes, num_steps)), np.zeros((num_episodes, num_steps)), np.zeros((num_episodes, num_steps))
    for episode in range(num_episodes):
        obs = env.reset()
        for step in range(num_steps):
            action, _states = policy.predict(obs)
            obs, reward, done, info = env.step(action)
            actions[episode, step] = action
            rewards[episode, step] = reward
            infos[episode, step] = info['objective']
            if done:
                break
    return actions, rewards, infos

