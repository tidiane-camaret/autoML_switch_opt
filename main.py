import os

from problem import NoisyHillsProblem, GaussianHillsProblem, RosenbrockProblem\
    ,RastriginProblem, SquareProblemClass, AckleyProblem, NormProblem, \
        YNormProblem
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


problemclass_train = YNormProblem
problemclass_test = YNormProblem

xlim = 2
nb_train_points = num_problems
nb_test_points = 100


train_starting_points = np.random.uniform(-xlim, xlim, size=(nb_train_points, 2))
train_problem_list = [problemclass_train(x0=xi) for xi in train_starting_points]
test_starting_points = np.random.uniform(-xlim, xlim, size=(nb_test_points, 2))
test_problem_list = [problemclass_test(x0=xi) for xi in test_starting_points]

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
        obs, _, _, info = test_env.step(action)
        actions_.append(action)
        obj_values_.append(info["obj_value"])
    actions.append(actions_)
    obj_values.append(obj_values_)

plt.plot(np.mean(obj_values, axis=0), label='untrained', alpha=0.7)
plt.fill_between(np.arange(len(obj_values[0])), np.mean(obj_values, axis=0) - np.std(obj_values, axis=0),
                 np.mean(obj_values, axis=0) + np.std(obj_values, axis=0), alpha=0.2)

policy.learn(total_timesteps=agent_training_timesteps,progress_bar=True, eval_freq=1000, eval_log_path='tb_logs/agent_eval')

#obj_values, np.array(trajectories), actions
#trained_actions, trained_rewards, = eval_agent(test_env, policy, num_steps=model_training_steps)
trained_obj_values, _ , trained_actions = eval_agent(test_env, policy, num_steps=model_training_steps)

plt.plot(np.mean(trained_obj_values, axis=0), label='trained', alpha=0.7)
plt.fill_between(np.arange(len(trained_obj_values[0])), np.mean(trained_obj_values, axis=0) - np.std(trained_obj_values, axis=0),
                 np.mean(trained_obj_values, axis=0) + np.std(trained_obj_values, axis=0), alpha=0.2)

# evaluate the handcrafted optimizers
for h_opt in optimizer_class_list:
    obj_values, trajectories = eval_handcrafted_optimizer(test_problem_list, h_opt, model_training_steps,
                                         do_init_weights=False, config=config)
    plt.plot(np.mean(obj_values, axis=0), label=h_opt.__name__, alpha=0.7, ls='--')
    plt.legend()
plt.show()


#print(trained_actions)
plt.plot(np.mean(actions, axis=0), label='actions')
plt.plot(np.mean(trained_actions, axis=0), label='trained_actions')
plt.legend()
plt.show()
