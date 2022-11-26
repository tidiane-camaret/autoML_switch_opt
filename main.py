import os

from problem import MLPProblemClass, SquareProblemClass
import torch
from environment import Environment
from eval_functions import eval_agent, eval_handcrafted_optimizer, eval_switcher_optimizer, first_index_below_threshold
import stable_baselines3
from eval_functions import eval_agent
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

starting_points = np.arange(-0.5, -0.25, 0.01)
train_problem_list = [SquareProblemClass(
    x0=x0
    ) for x0 in starting_points]

test_problem_list = [SquareProblemClass(x0=-0.4)]

# optimizer classes
optimizer_class_list = [torch.optim.SGD, torch.optim.Adam]
history_len = config.model.history_len

# define the environment based on the problem list
train_env = Environment(config=config,
                        problem_list=train_problem_list,
                        num_steps=model_training_steps,
                        history_len=history_len,
                        optimizer_class_list=optimizer_class_list,
                        
                        )
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
    policy = stable_baselines3.DQN('MlpPolicy', train_env, verbose=0, #exploration_fraction=config.policy.exploration_fraction,
                                   tensorboard_log='tb_logs/norm')
elif config.policy.model == 'PPO':
    policy = stable_baselines3.PPO('MlpPolicy', train_env, verbose=0,
                                   tensorboard_log='tb_logs/norm')
else:
    print('policy is not selected, it is set DQN')
    policy = stable_baselines3.DQN('MlpPolicy', train_env, verbose=0, exploration_fraction=config.policy.exploration_fraction,
                                   tensorboard_log='tb_logs/norm')
actions, obj_values = [], []
epochs = config.model.epochs
for _ in range(epochs):
    actions_, obj_values_ = [], []
    test_env.reset()
    for _ in range(model_training_steps):
        action = test_env.action_space.sample()
        obs, reward, _, info = test_env.step(action)
        actions_.append(action)
        obj_values_.append(info["obj_value"])
    actions.append(actions_)
    obj_values.append(obj_values_)

plt.plot(np.mean(obj_values, axis=0), label='untrained', alpha=0.7)
plt.fill_between(np.arange(len(obj_values[0])), np.mean(obj_values, axis=0) - np.std(obj_values, axis=0),
                 np.mean(obj_values, axis=0) + np.std(obj_values, axis=0), alpha=0.2)

policy.learn(total_timesteps=agent_training_timesteps)

#obj_values, np.array(trajectories), actions
#trained_actions, trained_rewards, = eval_agent(test_env, policy, num_steps=model_training_steps)
trained_rewards, _ , trained_actions = eval_agent(test_env, policy, num_steps=model_training_steps)

plt.plot(np.mean(trained_rewards, axis=0), label='trained', alpha=0.7)
plt.fill_between(np.arange(len(trained_rewards[0])), np.mean(trained_rewards, axis=0) - np.std(trained_rewards, axis=0),
                 np.mean(trained_rewards, axis=0) + np.std(trained_rewards, axis=0), alpha=0.2)

# evaluate the handcrafted optimizers
for h_opt in optimizer_class_list:
    rewards, trajectories = eval_handcrafted_optimizer(test_problem_list, h_opt, model_training_steps,
                                         do_init_weights=False, config=config)
    plt.plot(np.mean(rewards, axis=0), label=h_opt.__name__, alpha=0.7, ls='--')
    plt.legend()
plt.show()


#print(trained_actions)
plt.plot(np.mean(actions, axis=0), label='actions')
plt.plot(np.mean(trained_actions, axis=0), label='trained_actions')
plt.legend()
plt.show()
