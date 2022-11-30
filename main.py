import random

from problem import NoisyHillsProblem, GaussianHillsProblem, RosenbrockProblem\
    ,RastriginProblem, SquareProblemClass, AckleyProblem, NormProblem, \
        YNormProblem, MNISTProblemClass
import torch
from environment import Environment
from eval_functions import eval_agent, eval_handcrafted_optimizer
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
history_len = config.model.history_len


# define the problem list
nb_train_points = num_problems
nb_test_points = 100

### specific for math problems
math_problem_train_class = NoisyHillsProblem
math_problem_eval_class = GaussianHillsProblem

xlim = 2


if config.problem == 'MNIST':
  train_problem_list = [MNISTProblemClass(classes =[0,1]) for _ in range(num_problems)]
  test_problem_list = [MNISTProblemClass(classes = [2,3]) for _ in range(nb_test_points)]

elif config.problem == 'MathProblem':
  train_problem_list = [math_problem_train_class(x0=np.random.uniform(-xlim, xlim, size=(2))) 
                        for _ in range(num_problems)]
  test_problem_list = [math_problem_eval_class(x0=np.random.uniform(-xlim, xlim, size=(2))) 
                        for _ in range(nb_test_points)]


# optimizer classes
optimizer_class_list = [torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop]


# define the environment based on the problem list
reward_function = lambda x: -x
train_env = Environment(config=config,
                            problem_list=train_problem_list,
                            num_steps=model_training_steps,
                            history_len=history_len,
                            optimizer_class_list=optimizer_class_list,
                            reward_function=reward_function
                            )
# sanity check for the environment
check_env(train_env, warn=True)

# define the agent
if config.policy.model == 'PPO' or config.policy.optimization_mode == "soft":
    policy = stable_baselines3.PPO('MlpPolicy', train_env, verbose=0,)

elif config.policy.model == 'DQN':
    policy = stable_baselines3.DQN('MlpPolicy', train_env, verbose=0,
                                   exploration_fraction=config.policy.exploration_fraction,)

else:
    print('policy is not selected, it is set DQN')
    policy = stable_baselines3.DQN('MlpPolicy', train_env, verbose=0,
                                   exploration_fraction=config.policy.exploration_fraction,)



optimizers_trajectories = {}

# evaluate the handcrafted optimizers
for optimizer_class in optimizer_class_list:
    optimizer_name = optimizer_class.__name__
    optimizers_trajectories[optimizer_name] = {}
    obj_values, trajectories = eval_handcrafted_optimizer(test_problem_list, 
                                                            optimizer_class, 
                                                            model_training_steps, 
                                                            config, 
                                                            do_init_weights=False)

    optimizers_trajectories[optimizer_name]['obj_values'] = obj_values
    optimizers_trajectories[optimizer_name]['trajectories'] = trajectories

# train the agent

policy.learn(total_timesteps=agent_training_timesteps, 
            progress_bar=True,
            eval_freq=1000, 
            eval_log_path='tb_logs/'+config.problem+' '+config.policy.optimization_mode)

# evaluate the agent
train_env.train_mode = False # remove train mode, avoids calculating the lookahead

optimizers_trajectories['agent'] = {}

obj_values, trajectories, actions = eval_agent(train_env, 
                                                policy, 
                                                problem_list=test_problem_list, 
                                                num_steps=model_training_steps,
                                                random_actions=False, 
                                                )
optimizers_trajectories['agent']['obj_values'] = obj_values
optimizers_trajectories['agent']['trajectories'] = trajectories
optimizers_trajectories['agent']['actions'] = actions

# evaluate a random agent

optimizers_trajectories['random_agent'] = {}

obj_values, trajectories, actions = eval_agent(train_env, 
                                                policy, 
                                                problem_list=test_problem_list, 
                                                num_steps=model_training_steps,
                                                random_actions=False, 
                                                )
optimizers_trajectories['random_agent']['obj_values'] = obj_values
optimizers_trajectories['random_agent']['trajectories'] = trajectories
optimizers_trajectories['random_agent']['actions'] = actions



# plot mean objective value for each optimizer on the same plot
plt.figure(figsize=(10, 6))
for optimizer_name, optimizer_trajectories in optimizers_trajectories.items():
    obj_values = optimizer_trajectories['obj_values']
    plt.plot(np.mean(obj_values, axis=0), label=optimizer_name)
plt.legend()
plt.show()


# calculate a score matrix storing the mean objective value for each optimizer
score_matrix = np.zeros((nb_test_points, len(optimizers_trajectories.keys())))
for j, optimizer_name in enumerate(optimizers_trajectories.keys()):
    obj_values = optimizers_trajectories[optimizer_name]['obj_values']
    for i in range(nb_test_points):
        #score_matrix[i, j] = first_index_below_threshold(obj_values[i], threshold)
        score_matrix[i, j] = np.mean(obj_values[i][:])

# list of best optimizers for each starting point. if the agent is in a tie with a handcrafted optimizer, the agent wins
best_optimizer_list = []
for i in range(nb_test_points):
    best_score = np.inf
    best_optimizer = "agent"
    for j, optimizer_name in enumerate(optimizers_trajectories.keys()):
        if score_matrix[i, j] < best_score:
            best_score = score_matrix[i, j]
            best_optimizer = optimizer_name
    best_optimizer_list.append(best_optimizer)


# get indices where the agent is the best optimizer
plotted_starting_points = [i for i, x in enumerate(best_optimizer_list) if x == "agent"]
if len(plotted_starting_points) < 10:
    plotted_starting_points.extend(random.sample(range(nb_test_points), 10-len(plotted_starting_points)))


# for each starting point, plot the objective values of every optimizer
fig, axs = plt.subplots(2, 5, figsize=(20, 10))
for i, starting_point in enumerate(plotted_starting_points):
    ax = axs[i//5, i%5]
    for j, optimizer_name in enumerate(optimizers_trajectories.keys()):
        ax.plot(optimizers_trajectories[optimizer_name]['obj_values'][starting_point], label=optimizer_name)
    ax.set_title('starting point '+str(starting_point))
    ax.set_xlabel('iteration')
    ax.set_ylabel('objective value')
    ax.legend()
plt.show()






print("agent wins {} times".format(best_optimizer_list.count('agent')))

"""

plt.plot(np.mean(obj_values, axis=0), label='untrained', alpha=0.7)
plt.fill_between(np.arange(len(obj_values[0])), np.mean(obj_values, axis=0) - np.std(obj_values, axis=0),
                 np.mean(obj_values, axis=0) + np.std(obj_values, axis=0), alpha=0.2)

policy.learn(total_timesteps=agent_training_timesteps,progress_bar=True, eval_freq=1000, eval_log_path='tb_logs/agent_eval')

trained_rewards, _ , trained_actions = eval_agent(test_env, policy, num_steps=model_training_steps)

if config.policy.optimization_mode == 'soft':
  trained_beta1, trained_beta2 = trained_actions[0], trained_actions[1]
plt.plot(np.mean(trained_rewards, axis=0), label='trained', alpha=0.7)
plt.fill_between(np.arange(len(trained_rewards[0])), np.mean(trained_rewards, axis=0) - np.std(trained_rewards, axis=0),
                 np.mean(trained_rewards, axis=0) + np.std(trained_rewards, axis=0), alpha=0.2)

# evaluate the handcrafted optimizers
rewards_sgd, trajectories_sgd = eval_handcrafted_optimizer(test_problem_list, torch.optim.SGD, model_training_steps,
                                         do_init_weights=False, config=config)
rewards_adam, trajectories_adam = eval_handcrafted_optimizer(test_problem_list, torch.optim.Adam, model_training_steps,
                                          do_init_weights=False, config=config)
rewards_rmsprop, trajectories_rmsprop = eval_handcrafted_optimizer(test_problem_list, torch.optim.RMSprop, model_training_steps,
                                             do_init_weights=False, config=config)
plt.plot(np.mean(rewards_sgd, axis=0), label="SGD", alpha=0.7, color='red', ls='--')
plt.plot(np.mean(rewards_adam, axis=0), label="Adam", alpha=0.7, color='green', ls='--')
plt.plot(np.mean(rewards_rmsprop, axis=0), label="RMSprop", alpha=0.7, color='blue', ls='--')
plt.legend()
plt.show()
plt.savefig('graphs/eval.png')
plt.close()

#plt.plot(np.mean(actions[0], axis=0), label='actions')
if config.policy.optimization_mode == 'soft':
  plt.plot(np.mean(trained_beta1, axis=0), label='trained_actions')
  plt.legend()
  plt.show()
  plt.savefig('graphs/Beta1.png')
  plt.close()


  plt.plot(np.mean(trained_beta2, axis=0), label='trained_actions')
  plt.legend()
  plt.show()
  plt.savefig('graphs/Beta2.png')
  plt.close()

if config.policy.optimization_mode == 'hard':
  plt.plot(np.mean(actions, axis=0), label='actions')
  plt.plot(np.mean(trained_actions, axis=0), label='trained_actions')
  plt.legend()
  plt.show()
  plt.savefig('graphs/trained_actions.png')
  plt.close()
"""