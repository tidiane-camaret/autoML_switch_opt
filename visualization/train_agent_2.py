"""
evaluate an handcrafted and an agent based optimizer on a given problem list
extract the rewart, action and trajectory for each optimizer
"""

from problem import NoisyHillsProblem, GaussianHillsProblem, RosenbrockProblem\
    ,RastriginProblem, SquareProblemClass, AckleyProblem
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from eval_functions import eval_agent, eval_handcrafted_optimizer, eval_switcher_optimizer, first_index_below_threshold
import torch
from omegaconf import OmegaConf
from eval_functions import first_index_below_threshold
from environment import Environment
from stable_baselines3.common.env_checker import check_env
import stable_baselines3

# general parameters
config = OmegaConf.load('config.yaml')
num_agent_runs = config.model.num_agent_runs
model_training_steps = config.model.model_training_steps
agent_training_timesteps = num_agent_runs * model_training_steps
history_len = config.model.history_len

# list of handcrafted optimizers
optimizer_class_list = [torch.optim.SGD, torch.optim.Adam]

# define the problem lists
xlim = 1
nb_train_points = 3
nb_test_points = 3
problemclass = AckleyProblem

train_starting_points = np.random.uniform(-xlim, xlim, size=(nb_train_points, 2))
train_problem_list = [problemclass(x0=xi) for xi in train_starting_points]
test_starting_points = np.random.uniform(-xlim, xlim, size=(nb_test_points, 2))
test_problem_list = [problemclass(x0=xi) for xi in test_starting_points]

# meshgrid for plotting the problem surface
x = np.arange(-xlim, xlim, xlim / 100)
y = np.arange(-xlim, xlim, xlim / 100)
X, Y = np.meshgrid(x, y)
X, Y = torch.tensor(X), torch.tensor(Y)
Z = problemclass().function_def(X, Y)
Z = Z.detach().numpy()


# parameters of the environment
reward_function = lambda x: -x 
train_env = Environment(config=config,
                        problem_list=train_problem_list,
                        num_steps=model_training_steps,
                        history_len=history_len,
                        optimizer_class_list=optimizer_class_list,
                        reward_function=reward_function
                        
                        )
check_env(train_env, warn=True)

policy = stable_baselines3.DQN('MlpPolicy', train_env, verbose=0, exploration_fraction=config.policy.exploration_fraction,
                                   tensorboard_log='tb_logs/single_problem_gaussian_hills_dqn')

# define the results dictionary
results = {}

# evaluate the handcrafted optimizers
for optimizer_class in optimizer_class_list:
    optimizer_name = optimizer_class.__name__
    results[optimizer_name] = {}
    obj_values, trajectories = eval_handcrafted_optimizer(test_problem_list, 
                                                            optimizer_class, 
                                                            model_training_steps, 
                                                            config, 
                                                            do_init_weights=False)

    results[optimizer_name]['obj_values'] = obj_values
    results[optimizer_name]['trajectories'] = trajectories


# evaluate a switcher optimizer
for switch_time in [0.5]:
    optimizer_name = 'switcher_' + str(switch_time)
    results[optimizer_name] = {}
    obj_values, trajectories = eval_handcrafted_optimizer(test_problem_list, 
                                                            optimizer_class_list[0],
                                                            model_training_steps, 
                                                            config, 
                                                            do_init_weights=False,
                                                            optimizer_2_class=optimizer_class_list[0],
                                                            switch_time=switch_time)
                                                

    results[optimizer_name]['obj_values'] = obj_values
    results[optimizer_name]['trajectories'] = trajectories

# train and evaluate the agent
policy.learn(total_timesteps=10, progress_bar=True,eval_freq=1000, eval_log_path='tb_logs/single_problem_gaussian_hills_dqn_eval')
optimizer_name = 'agent'
results[optimizer_name] = {}
obj_values, trajectories, actions = eval_agent(train_env, 
                                                policy, 
                                                problem_list=test_problem_list, 
                                                num_steps=model_training_steps, 
                                                )
results[optimizer_name]['obj_values'] = obj_values
results[optimizer_name]['trajectories'] = trajectories
results[optimizer_name]['actions'] = actions




# for each starting point, define best optimizer as the one with the lowest final objective value
best_optimizer_list = []
for i in range(nb_test_points):
    best_optimizer = None
    best_obj_value = np.inf
    for optimizer_name in results.keys():
        obj_values = results[optimizer_name]['obj_values']
        if obj_values[i][-1] < best_obj_value:
            best_obj_value = obj_values[i][-1]
            best_optimizer = optimizer_name
    best_optimizer_list.append(best_optimizer)

# plot the best optimizer for each starting point on the problem surface. different color for each optimizer
fig, ax = plt.subplots()
for i in range(nb_test_points):
    optimizer_name = best_optimizer_list[i]
    obj_values = results[optimizer_name]['obj_values']
    trajectories = results[optimizer_name]['trajectories']
    ax.plot(trajectories[i][:,0], trajectories[i][:,1], label=optimizer_name)
ax.contour(X, Y, Z, levels=20, norm=LogNorm())
ax.legend()
plt.show()

"""

# display result trajectories of each optimizer in a separate subplot
optimizer_names = list(results.keys())
fig, axs = plt.subplots(1, len(optimizer_names), figsize=(15, 5))
for i, optimizer_name in enumerate(optimizer_names):
    trajectories = results[optimizer_name]['trajectories']
    for trajectory in trajectories:
        axs[i].plot(trajectory[:, 0], trajectory[:, 1], 'o-', c='b', alpha=0.5, markersize=2)
    axs[i].set_title(optimizer_name)
    axs[i].legend()
    axs[i].contourf(X, Y, Z)

    '''
    axs[i].set_xlim(-xlim, xlim)
    axs[i].set_ylim(-xlim, xlim)
    axs[i].contour(X, Y, Z, 20, cmap='RdGy', norm=LogNorm(vmin=1.0, vmax=1000.0))
    axs[i].set_aspect('equal', 'box')
    '''
plt.show()

"""