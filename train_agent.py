import numpy as np




def eval_agent(env, policy, num_episodes=1, num_steps=5):
    actions, obj_values = np.zeros((num_episodes, num_steps)), np.zeros((num_episodes, num_steps))
    for episode in range(num_episodes):
        obs = env.reset()
        for step in range(num_steps):
            action, _states = policy.predict(obs)
            obs, reward, done, info = env.step(action)
            actions[episode, step] = action
            obj_values[episode, step] = info["obj_value"]
            if done:
                break
    return actions, obj_values

def eval_random_agent(env, num_episodes=1, num_steps=5):
    actions, obj_values = np.zeros((num_episodes, num_steps)), np.zeros((num_episodes, num_steps))
    for episode in range(num_episodes):
        obs = env.reset()
        for step in range(num_steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            actions[episode, step] = action
            obj_values[episode, step] = info["obj_value"]
            if done:
                break
    return actions, obj_values