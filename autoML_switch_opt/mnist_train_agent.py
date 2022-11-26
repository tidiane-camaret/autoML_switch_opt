# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 13:10:57 2022

@author: DELL
"""

import os

from problem import *
import torch
from environment import *
import stable_baselines3
from train_agent import eval_agent
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from omegaconf import OmegaConf

config = OmegaConf.load('config.yaml')
# number of problems to train on
num_problems = config.model.num_problems

# number of agent training episodes
num_agent_runs = config.model.num_agent_runs
# number of steps in each episode
model_training_steps = config.model.model_training_steps

agent_training_timesteps = num_agent_runs * model_training_steps
# define the problem list
train_problem_list = [MNISTProblemClass(classes =[0,1]) for _ in range(num_problems)]
test_problem_list = [MNISTProblemClass(classes = [2,3])]

# optimizer classes
optimizer_class_list = [torch.optim.SGD, torch.optim.RMSprop, torch.optim.Adam]
history_len = config.model.history_len

# define the environment based on the problem list
train_env = Environment(config=config,
                        problem_list=train_problem_list,
                        num_steps=model_training_steps,
                        history_len=history_len,
                        optimizer_class_list=optimizer_class_list,
                        do_init_weights=False
                        )
#test_env = train_env
test_env = Environment(config=config,
                        problem_list=test_problem_list,
                        num_steps=model_training_steps,
                        history_len=history_len,
                        optimizer_class_list=optimizer_class_list,
                        do_init_weights=False
                        )

# sanity check for the environment
check_env(train_env, warn=True)
check_env(test_env, warn=True)

# define the agent
if config.policy.model == 'DQN':
    policy = stable_baselines3.DQN('MlpPolicy', train_env, verbose=0, exploration_fraction=config.policy.exploration_fraction,
                                   tensorboard_log='tb_logs/norm')
elif config.policy.model == 'PPO':
    policy = stable_baselines3.PPO('MlpPolicy', train_env, verbose=0,
                                   tensorboard_log='tb_logs/norm')
else:
    print('policy is not selected, it is set DQN')
    policy = stable_baselines3.DQN('MlpPolicy', train_env, verbose=0, exploration_fraction=config.policy.exploration_fraction,
                                   tensorboard_log='tb_logs/norm')
actions, rewards = [], []
epochs = config.model.epochs
for _ in range(epochs):
    actions_, rewards_ = [], []
    test_env.reset()
    for _ in range(model_training_steps):
        action = test_env.action_space.sample()
        obs, reward, _, _ = test_env.step(action)
        actions_.append(action)
        rewards_.append(reward)
    actions.append(actions_)
    rewards.append(rewards_)

plt.plot(np.mean(rewards, axis=0), label='untrained', alpha=0.7)
plt.fill_between(np.arange(len(rewards[0])), np.mean(rewards, axis=0) - np.std(rewards, axis=0),
                 np.mean(rewards, axis=0) + np.std(rewards, axis=0), alpha=0.2)

policy.learn(total_timesteps=agent_training_timesteps)

trained_actions, trained_rewards = eval_agent(test_env, policy, num_episodes=10, num_steps=model_training_steps)

plt.plot(np.mean(trained_rewards, axis=0), label='trained', alpha=0.7)
plt.fill_between(np.arange(len(trained_rewards[0])), np.mean(trained_rewards, axis=0) - np.std(trained_rewards, axis=0),
                 np.mean(trained_rewards, axis=0) + np.std(trained_rewards, axis=0), alpha=0.2)

# evaluate the handcrafted optimizers
rewards_sgd = eval_handcrafted_optimizer(test_problem_list, torch.optim.SGD, model_training_steps,
                                         do_init_weights=False, config=config)
rewards_adam = eval_handcrafted_optimizer(test_problem_list, torch.optim.Adam, model_training_steps,
                                          do_init_weights=False,config=config)
rewards_rmsprop = eval_handcrafted_optimizer(test_problem_list, torch.optim.RMSprop, model_training_steps,
                                             do_init_weights=False, config=config)
plt.plot(np.mean(rewards_sgd, axis=0), label="SGD", alpha=0.7, color='red', ls='--')
plt.plot(np.mean(rewards_adam, axis=0), label="Adam", alpha=0.7, color='green', ls='--')
plt.plot(np.mean(rewards_rmsprop, axis=0), label="RMSprop", alpha=0.7, color='blue', ls='--')
plt.legend()
plt.show()
print(trained_actions)
plt.plot(np.mean(actions, axis=0), label='actions')
plt.plot(np.mean(trained_actions, axis=0), label='trained_actions')
plt.legend()
plt.show()
