"""
evaluate an handcrafted and an agent based optimizer on a given problem list
extract the rewart, action and trajectory for each optimizer
"""
import os, pickle
from problem import NoisyHillsProblem, GaussianHillsProblem, RosenbrockProblem\
    ,RastriginProblem, SquareProblemClass, AckleyProblem, NormProblem, \
        YNormProblem
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from eval_functions import eval_agent, eval_handcrafted_optimizer
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
xlim = 2
nb_train_points = config.model.num_problems
nb_test_points = 100


def train_and_eval_agent(problemclass1, problemclass2, agent_training_timesteps, do_plot=True):

    train_starting_points = np.random.uniform(-xlim, xlim, size=(nb_train_points, 2))
    train_problem_list = [problemclass1(x0=xi) for xi in train_starting_points]
    test_starting_points = np.random.uniform(-xlim, xlim, size=(nb_test_points, 2))
    test_problem_list = [problemclass2(x0=xi) for xi in test_starting_points]

    # meshgrid for plotting the problem surface
    x = np.arange(-xlim, xlim, xlim / 100)
    y = np.arange(-xlim, xlim, xlim / 100)
    X, Y = np.meshgrid(x, y)
    X, Y = torch.tensor(X), torch.tensor(Y)
    Z = problemclass2().function_def(X, Y)
    Z = Z.detach().numpy()

    # calculate minimum of the problem surface
    # and determine the threshold for the reward function
    function_min = np.min(Z)
    print('test function minimum: ', function_min)
    threshold = function_min + 0.001



    # parameters of the environment
    reward_function = lambda x: 10 if x < threshold else -1
    train_env = Environment(config=config,
                            problem_list=train_problem_list,
                            num_steps=model_training_steps,
                            history_len=history_len,
                            optimizer_class_list=optimizer_class_list,
                            reward_function=reward_function
                            
                            )
    check_env(train_env, warn=True)

    policy = stable_baselines3.DQN('MlpPolicy', train_env, verbose=0, #exploration_fraction=config.policy.exploration_fraction,
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
    policy.learn(total_timesteps=agent_training_timesteps, progress_bar=True,eval_freq=1000, eval_log_path='tb_logs/single_problem_gaussian_hills_dqn_eval')
    optimizer_name = 'agent'
    results[optimizer_name] = {}
    train_env.train_mode = False # remove train mode, avoids calculating the lookahead
    obj_values, trajectories, actions = eval_agent(train_env, 
                                                    policy, 
                                                    problem_list=test_problem_list, 
                                                    num_steps=model_training_steps, 
                                                    )
    results[optimizer_name]['obj_values'] = obj_values
    results[optimizer_name]['trajectories'] = trajectories
    results[optimizer_name]['actions'] = actions

    params_dict = {
        'test_starting_points': test_starting_points,
        'threshold': threshold,
        'meshgrid': (X, Y, Z),

    }

    return results, params_dict

def agent_statistics(results, params_dict, do_plot=True):
    
    test_starting_points = params_dict['test_starting_points']
    threshold = params_dict['threshold']
    X, Y, Z = params_dict['meshgrid']
    
    #### DISPLAY RESULTS ####

    # calculate a score matrix storing the mean objective value for each optimizer
    score_matrix = np.zeros((nb_test_points, len(results.keys())))
    for i in range(nb_test_points):
        for j, optimizer_name in enumerate(results.keys()):
            obj_values = results[optimizer_name]['obj_values']
            #score_matrix[i, j] = first_index_below_threshold(obj_values[i], threshold)
            score_matrix[i, j] = np.mean(obj_values[i][:])

    # list of best optimizers for each starting point. if the agent is in a tie with a handcrafted optimizer, the agent wins
    best_optimizer_list = []
    for i in range(nb_test_points):
        best_score = np.inf
        best_optimizer = "agent"
        for j, optimizer_name in enumerate(results.keys()):
            if score_matrix[i, j] < best_score:
                best_score = score_matrix[i, j]
                best_optimizer = optimizer_name
        best_optimizer_list.append(best_optimizer)

    # plot matrix values as lines, sorted by best optimizer
    score_matrix_sorted = score_matrix[score_matrix[:, -1].argsort()]

    fig, ax = plt.subplots(1,2,figsize=(10, 10))
    for i, optimizer_name in enumerate(results.keys()):
        ax[0].plot(score_matrix_sorted[:, i], label=optimizer_name, alpha=0.7)
    ax[0].legend()
    ax[0].set_xlabel('starting point')
    ax[0].set_ylabel('score')
    ax[0].set_title('score for each starting point and optimizer')


    # plot the best optimizer for each starting point on the problem surface. 

    ax[1].contourf(X, Y, Z, 50, cmap="gray")
    ax[1].set_title('Best optimizer for each starting point')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    #loop through unique best optimizers
    for optimizer_name in np.unique(best_optimizer_list):
        # get the starting points for which the optimizer is the best
        idx = np.array(best_optimizer_list) == optimizer_name
        # plot the starting points
        ax[1].scatter(test_starting_points[idx, 0], test_starting_points[idx, 1], label=optimizer_name)
    ax[1].legend()

    if do_plot:
        plt.show()

    # plot the objective values for each starting point and optimizer
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(nb_test_points):
        optimizer_name = best_optimizer_list[i]
        trajectories = results[optimizer_name]['trajectories']
        #ax.plot(trajectories[i][:, 0], trajectories[i][:, 1], 'o-', c='b', alpha=0.5, markersize=1)
    ax.legend()
    plt.savefig("visualization/graphs/best_opt.png")
    if do_plot:
        plt.show()
    """
    #analyze the actions taken by the agent
    optimizer_name = "agent"
    actions = results[optimizer_name]['actions']
    trajectories = results[optimizer_name]['trajectories']

    # put all actions in a single array and plot the matrix 
    fig, ax = plt.subplots(1,2,figsize=(10, 10))
    ax[0].imshow(actions)
    ax[0].set_title('actions taken by the agent')
    ax[0].set_xlabel('step')
    ax[0].set_ylabel('starting point')


    # on the problem surface, plot agent actions for each starting point
    ax[1].contourf(X, Y, Z, 50, cmap="gray")
    ax[1].set_title('Agent actions for each starting point')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    for i in range(nb_test_points):
        ax[1].scatter(trajectories[i][:, 0], trajectories[i][:, 1], c=actions[i,:], s=1)
    plt.savefig("visualization/graphs/agent_actions.png")
    if do_plot:
        plt.show()

    optimizer_names = list(results.keys())  
    """
    # display result trajectories of each optimizer in a separate subplot
    
    fig, axs = plt.subplots(1, len(optimizer_names), figsize=(15, 5))
    for i, optimizer_name in enumerate(optimizer_names):
        trajectories = results[optimizer_name]['trajectories']
        for j in range(nb_test_points):
            if best_optimizer_list[j] == optimizer_name:
                axs[i].plot(trajectories[j][:, 0], trajectories[j][:, 1], 'o-', c='b', alpha=0.5, markersize=1)
        axs[i].set_title(optimizer_name)
        axs[i].legend()
        axs[i].contourf(X, Y, Z)


    plt.savefig("visualization/graphs/trajectories.png")
    if do_plot:
        plt.show()
    """

    # select a starting point where the agent is the best optimizer
    idx = np.array(best_optimizer_list) == "agent"
    idx = np.where(idx)[0][0]
    
    # plot the trajectories of all optimizers for the selected starting point
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].contourf(X, Y, Z, 50, cmap="gray")
    ax[0].set_title('Trajectories for the selected starting point')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    for i, optimizer_name in enumerate(optimizer_names):
        trajectories = results[optimizer_name]['trajectories']
        ax[0].plot(trajectories[idx][:, 0], trajectories[idx][:, 1], 'o-', label=optimizer_name)
    ax[0].legend()

    
    # plot the objective values for the selected starting point and optimizer
    ax[1].set_title('Objective values for the selected starting point')
    ax[1].set_xlabel('step')
    ax[1].set_ylabel('objective value')
    for i, optimizer_name in enumerate(optimizer_names):
        obj_values = results[optimizer_name]['obj_values']
        ax[1].plot(obj_values[idx], label=optimizer_name)
    ax[1].legend()
    plt.savefig("visualization/graphs/objective_values_selected.png")
    if do_plot:
        plt.show()
    # count the number of times each optimizer is the best
    best_optimizer_count = {}
    for optimizer_name in optimizer_names:
        best_optimizer_count[optimizer_name] = np.sum(np.array(best_optimizer_list) == optimizer_name)


    plt.close('all')
    return best_optimizer_count, score_matrix

if __name__ == "__main__":

    problemclass1 = GaussianHillsProblem
    problemclass2 = GaussianHillsProblem

    filename = "visualization/graphs/"\
                + problemclass1.__name__\
                + "_" + problemclass2.__name__\
                + "_results.pkl"

    #if file already exists, load it
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            results, params_dict = pickle.load(f)

    #else, run the experiment and save the results
    else:
        results, params_dict = train_and_eval_agent(problemclass1, problemclass2, agent_training_timesteps)
        with open(filename, 'wb') as f:
            pickle.dump((results, params_dict), f)

    best_optimizer_count, score_matrix = agent_statistics(results, params_dict, do_plot=True)
    print(best_optimizer_count)