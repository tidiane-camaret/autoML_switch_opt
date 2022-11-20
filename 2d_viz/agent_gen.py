import sys
sys.path.append('..')
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
xlim = 2


nb_train_points = 200
nb_test_points = nb_train_points

optimizer_class_list = [torch.optim.SGD,torch.optim.Adam]


config = OmegaConf.load('../config.yaml')
num_agent_runs = config.model.num_agent_runs
# number of steps in each episode
model_training_steps = config.model.model_training_steps
agent_training_timesteps = num_agent_runs * model_training_steps
history_len = config.model.history_len

# generate list of random starting points
starting_points = np.random.uniform(-xlim, xlim, size=(nb_train_points, 2))
train_problem_list = [AckleyProblem(x0=xi) for xi in starting_points]
starting_points = np.random.uniform(-xlim, xlim, size=(nb_test_points, 2))
test_problem_list = [AckleyProblem(x0=xi) for xi in starting_points]



last_values = []

for optimizer in optimizer_class_list:
    loss_curves, trajectories = eval_handcrafted_optimizer(test_problem_list, optimizer, model_training_steps, config, do_init_weights=False)

    #fibt = np.array([first_index_below_threshold(lc, threshold) for lc in loss_curves])
    
    # plot surface using starting points and fibt
    plt.scatter(starting_points[:,0], starting_points[:,1], c=loss_curves[:,-1], cmap='viridis')
    plt.colorbar()
    plt.show()#plt.savefig(optimizer.__name__ + "_eval.png")

    print(loss_curves[:,-1])
    last_values.append(loss_curves[:,-1])

loss_curves = eval_switcher_optimizer(test_problem_list, 
                        optimizer_class_list, 
                        model_training_steps, 
                        config, 
                        switch_time=0.5, 
                        do_init_weights=False)

last_values.append(loss_curves[:,-1])


# define the environment based on the problem list
train_env = Environment(config=config,
                        problem_list=train_problem_list,
                        num_steps=model_training_steps,
                        history_len=history_len,
                        optimizer_class_list=optimizer_class_list,
                        #reward_function=reward_function
                        
                        )
check_env(train_env, warn=True)

policy = stable_baselines3.DQN('MlpPolicy', train_env, verbose=0, exploration_fraction=config.policy.exploration_fraction,
                                   tensorboard_log='tb_logs/gaussian_hills_dqn')

agent_scores = []
nb_training_seqs = 10

for i in range(nb_training_seqs):


    policy.learn(total_timesteps=agent_training_timesteps, 
                progress_bar=True,
                tb_log_name='gaussian_hills_dqn')
    policy.save('models/gaussian_hills_dqn')
    trained_actions, trained_obj_values = eval_agent(train_env, 
                                        policy, 
                                        problem_list=test_problem_list, 
                                        num_steps=model_training_steps, 
                                        )

    #print(len(last_values))
    all_last_values = last_values + [trained_obj_values[:,-1]]
    print(np.array(all_last_values).shape)
    best_optimizer = np.argmin(
        np.array(all_last_values), 
        axis=0)

    optimizer_name_list = [optimizer.__name__ for optimizer in optimizer_class_list] + ['switcher', 'agent']
    # plot best optimizer for each starting point, with legend

    fig, ax = plt.subplots()

    for b in np.unique(best_optimizer):
        ax.scatter(starting_points[best_optimizer==b,0], starting_points[best_optimizer==b,1], label=optimizer_name_list[b])
    ax.legend()
    fig.savefig("graphs/best_opt_"+str(i)+".png")
    agent_score = np.mean(best_optimizer == len(optimizer_name_list)-1)
    print("agent score: ", agent_score)
    agent_scores.append(agent_score)

# plot agent score over time
plt.plot(agent_scores)
plt.show()#savefig("agent_score.png")









