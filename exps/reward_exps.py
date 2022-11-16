import sys
sys.path.append('..')


from problem import MLPProblemClass, RosenbrockProblemClass, SquareProblemClass
import torch
from environment import Environment, eval_handcrafted_optimizer
import stable_baselines3
from train_agent import eval_agent, eval_random_agent
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from omegaconf import OmegaConf
from problem_evaluation import first_index_below_threshold


config = OmegaConf.load('../config.yaml')
# number of problems to train on
num_problems = config.model.num_problems
# number of agent training episodes
num_agent_runs = config.model.num_agent_runs
# number of steps in each episode
model_training_steps = config.model.model_training_steps

agent_training_timesteps = num_agent_runs * model_training_steps
# define the problem list

train_starting_points = np.arange(-1, 1, 0.01)

train_problem_list = [SquareProblemClass(
    x0=x0
    ) for x0 in train_starting_points]

test_starting_points = np.arange(-1, 1, 0.1)

test_problem_list = [SquareProblemClass(
    x0=x0
    ) for x0 in test_starting_points]

# optimizer classes
optimizer_class_list = [torch.optim.SGD, torch.optim.Adam]
history_len = config.model.history_len


threshold = 0.05
reward_function = lambda x: 100 if x < threshold else -1

# define the environment based on the problem list
train_env = Environment(config=config,
                        problem_list=train_problem_list,
                        num_steps=model_training_steps,
                        history_len=history_len,
                        optimizer_class_list=optimizer_class_list,
                        reward_function=reward_function
                        
                        )
test_env = Environment(config=config,
                       problem_list=test_problem_list,
                       num_steps=model_training_steps,
                       history_len=history_len,
                       optimizer_class_list=optimizer_class_list,
                       do_init_weights=False,
                       reward_function=reward_function
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
actions, obj_values = [], []
epochs = config.model.epochs

policy.learn(total_timesteps=agent_training_timesteps)

actions, obj_values = eval_agent(test_env, 
                                policy, 
                                problem_list=test_problem_list, 
                                num_steps=model_training_steps, 
                                random_actions=True
                                )

trained_actions, trained_obj_values = eval_agent(test_env, 
                                policy, 
                                problem_list=test_problem_list, 
                                num_steps=model_training_steps, 
                                )

print(np.apply_along_axis(first_index_below_threshold, 1, trained_obj_values, threshold))
# for each trajectory, plot its first time reaching the threshold
plt.plot(test_starting_points, np.apply_along_axis(first_index_below_threshold, 1, trained_obj_values, threshold), label='trained')
plt.plot(test_starting_points, np.apply_along_axis(first_index_below_threshold, 1, obj_values, threshold), label='random')


# evaluate the handcrafted optimizers
for h_opt in optimizer_class_list:
    handcrafted_obj_values, trajectories = eval_handcrafted_optimizer(test_problem_list, h_opt, model_training_steps,
                                         do_init_weights=False, config=config)
    plt.plot(test_starting_points, np.apply_along_axis(first_index_below_threshold, 1, handcrafted_obj_values, threshold), label=h_opt.__name__, alpha=0.7, ls='--')
    plt.legend()
plt.show()


#print(trained_actions)
plt.plot(np.mean(actions, axis=0), label='actions')
plt.plot(np.mean(trained_actions, axis=0), label='trained_actions')
plt.legend()
plt.show()

plt.matshow(trained_actions, label='trained_actions')
plt.legend()
plt.show()





# plot the mean trajectories of handcrafted, random and trained optimizers

for h_opt in optimizer_class_list:
    obj_values, trajectories = eval_handcrafted_optimizer(test_problem_list, h_opt, model_training_steps,
                                         do_init_weights=False, config=config)
    plt.plot(np.mean(obj_values, axis=0), label=h_opt.__name__, alpha=0.7, ls='--')

plt.plot(np.mean(obj_values, axis=0), label='untrained', alpha=0.7)
plt.fill_between(np.arange(len(obj_values[0])), np.mean(obj_values, axis=0) - np.std(obj_values, axis=0),
                 np.mean(obj_values, axis=0) + np.std(obj_values, axis=0), alpha=0.2)
                

plt.plot(np.mean(trained_obj_values, axis=0), label='trained', alpha=0.7)
plt.fill_between(np.arange(len(trained_obj_values[0])), np.mean(trained_obj_values, axis=0) - np.std(trained_obj_values, axis=0),
                 np.mean(trained_obj_values, axis=0) + np.std(trained_obj_values, axis=0), alpha=0.2)
plt.legend()
plt.show()