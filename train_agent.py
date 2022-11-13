from environment import Environment
from problem import mlp_problem, MLPProblemClass
import torch 
from stable_baselines3.common.env_checker import check_env
import stable_baselines3
import numpy as np  
import matplotlib.pyplot as plt
from environment import eval_handcrafted_optimizer


def eval_agent(env, policy, num_episodes=1, num_steps=5):
    actions, rewards = np.zeros((num_episodes, num_steps)), np.zeros((num_episodes, num_steps))
    for episode in range(num_episodes):
        obs = env.reset()
        for step in range(num_steps):
            action, _states = policy.predict(obs)
            obs, reward, done, info = env.step(action)
            actions[episode, step] = action
            rewards[episode, step] = reward
            if done:
                break
    return actions, rewards

# number of problems to train on
num_problems = 100
# number of agent training episodes
num_agent_runs = 100
# number of steps in each episode
model_training_steps = 100

agent_training_timesteps = num_agent_runs * model_training_steps
# define the problem list
train_problem_list = [MLPProblemClass() for _ in range(num_problems)]
test_problem_list = [MLPProblemClass()]

# optimizer classes
optimizer_class_list=[torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop]
history_len = 25

# define the environment based on the problem list
train_env = Environment(problem_list = train_problem_list,
                num_steps = model_training_steps, 
                history_len = history_len, 
                optimizer_class_list = optimizer_class_list
                )
test_env = Environment(problem_list = test_problem_list,
                num_steps = model_training_steps, 
                history_len = history_len, 
                optimizer_class_list = optimizer_class_list,
                do_init_weights = False
                )


# sanity check for the environment
check_env(train_env, warn=True)

# define the agent
policy = stable_baselines3.DQN('MlpPolicy', train_env, verbose=0,
                                         tensorboard_log='tb_logs/norm')


actions, rewards = [], []
for _ in range(10):
  actions_,rewards_ = [], []
  test_env.reset()
  for _ in range(model_training_steps):
    action = test_env.action_space.sample()
    obs, reward, _, _ = test_env.step(action)
    actions_.append(action)
    rewards_.append(reward)
  actions.append(actions_)
  rewards.append(rewards_)


plt.plot(np.mean(rewards, axis=0), label='untrained', alpha=0.7)
plt.fill_between(np.arange(len(rewards[0])), np.mean(rewards, axis=0) - np.std(rewards, axis=0), np.mean(rewards, axis=0) + np.std(rewards, axis=0), alpha=0.2)


policy.learn(total_timesteps=agent_training_timesteps)


trained_actions, trained_rewards = eval_agent(test_env, policy, num_episodes=10, num_steps=model_training_steps)

plt.plot(np.mean(trained_rewards, axis=0), label='trained', alpha=0.7)
plt.fill_between(np.arange(len(trained_rewards[0])), np.mean(trained_rewards, axis=0) - np.std(trained_rewards, axis=0), np.mean(trained_rewards, axis=0) + np.std(trained_rewards, axis=0), alpha=0.2)


#evaluate the handcrafted optimizers
rewards_sgd = eval_handcrafted_optimizer(test_problem_list, torch.optim.SGD, model_training_steps, do_init_weights=False)
rewards_adam = eval_handcrafted_optimizer(test_problem_list, torch.optim.Adam, model_training_steps, do_init_weights=False)
rewards_rmsprop = eval_handcrafted_optimizer(test_problem_list, torch.optim.RMSprop, model_training_steps, do_init_weights=False)
plt.plot(np.mean(rewards_sgd,axis=0), label="SGD", alpha=0.7, color='red', ls = '--')
plt.plot(np.mean(rewards_adam,axis=0), label="Adam" , alpha=0.7, color='green', ls = '--')
plt.plot(np.mean(rewards_rmsprop,axis=0), label="RMSprop", alpha=0.7, color='blue', ls = '--')
plt.legend()
plt.show()


plt.plot(np.mean(actions, axis=0), label='actions')
plt.plot(np.mean(trained_actions, axis=0), label='trained_actions')
plt.legend()
plt.show()
