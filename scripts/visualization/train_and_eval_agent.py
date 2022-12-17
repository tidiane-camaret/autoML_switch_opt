"""
evaluate an handcrafted and an agent based optimizer on a given problem list
extract the rewart, action and trajectory for each optimizer
"""
import os, pickle
from problem import NoisyHillsProblem, GaussianHillsProblem,\
     RastriginProblem, AckleyProblem, NormProblem

import numpy as np
from eval_functions import eval_agent, eval_handcrafted_optimizer
import torch
from omegaconf import OmegaConf
from eval_functions import first_index_below_threshold
from environment import Environment
from stable_baselines3.common.env_checker import check_env
import stable_baselines3
import random
import copy

# general parameters
config = OmegaConf.load('config.yaml')
num_agent_runs = config.model.num_agent_runs
model_training_steps = config.model.model_training_steps
agent_training_timesteps = num_agent_runs * model_training_steps
history_len = config.model.history_len

# list of handcrafted optimizers
optimizer_class_list = [torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop]
all_problems_class_list = [NoisyHillsProblem, GaussianHillsProblem, RastriginProblem, AckleyProblem, NormProblem]
# define the problem lists
xlim = 2
nb_train_points = config.model.num_problems
nb_test_points = 100
history_len = config.model.history_len
exploration_fraction = config.policy.exploration_fraction
lr = config.model.lr
reward_system = config.environment.reward_system
optimization_mode = config.policy.optimization_mode


def train_and_eval_agent(problemclass_train, problemclass_eval, agent_training_timesteps, do_plot=True):
    
    if problemclass_train == "all_except_eval":
        #problemclass_train_list contains all problems except the one to be evaluated
        problemclass_train_list = copy.deepcopy(all_problems_class_list)
        problemclass_train_list.remove(problemclass_eval)
        print("problemclass_train_list: ", problemclass_train_list)
        train_problem_list = [random.choice(problemclass_train_list)(x0=np.random.uniform(-xlim, xlim, size=(2))) for _ in range(nb_train_points)]
    
    elif problemclass_train == "none":
        # if none, pick a class at random. will not be used for training
        train_starting_points = np.random.uniform(-xlim, xlim, size=(nb_train_points, 2))
        train_problem_list = [NoisyHillsProblem(x0=xi) for xi in train_starting_points]
    else : 
        train_starting_points = np.random.uniform(-xlim, xlim, size=(nb_train_points, 2))
        train_problem_list = [problemclass_train(x0=xi) for xi in train_starting_points]

    test_starting_points = np.random.uniform(-xlim, xlim, size=(nb_test_points, 2))
    test_problem_list = [problemclass_eval(x0=xi) for xi in test_starting_points]

    # meshgrid for plotting the problem surface
    x = np.arange(-xlim, xlim, xlim / 100)
    y = np.arange(-xlim, xlim, xlim / 100)
    X, Y = np.meshgrid(x, y)
    X, Y = torch.tensor(X), torch.tensor(Y)
    Z = problemclass_eval().function_def(X, Y)
    Z = Z.detach().numpy()

    # calculate minimum of the problem surface
    # and determine the threshold for the reward function
    function_min = np.min(Z)
    #print('test function minimum: ', function_min)
    threshold = function_min + 0.001



    # parameters of the environment
    train_env = Environment(
                            problem_list=train_problem_list,
                            num_steps=model_training_steps,
                            history_len=history_len,
                            optimization_mode=optimization_mode,
                            lr=lr,
                            reward_system=reward_system,
                            threshold=threshold,
                            optimizer_class_list=optimizer_class_list,
                            )
                        
    check_env(train_env, warn=True)

    # define the agent
    if config.policy.model == 'PPO' or config.policy.optimization_mode == "soft":
        policy = stable_baselines3.PPO('MlpPolicy', train_env, verbose=0,
                                    tensorboard_log='results/tb_logs/norm')

    elif config.policy.model == 'DQN':
        policy = stable_baselines3.DQN('MlpPolicy', train_env, verbose=0,
                                    exploration_fraction=config.policy.exploration_fraction,
                                    tensorboard_log='results/tb_logs/norm')

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
    """
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
    """
    # train and evaluate the agent
    if problemclass_train != "none":
        policy.learn(total_timesteps=agent_training_timesteps, progress_bar=True,eval_freq=1000, eval_log_path='tb_logs/single_problem_gaussian_hills_dqn_eval')
    optimizer_name = 'agent'
    results[optimizer_name] = {}
    train_env.train_mode = False # remove train mode, avoids calculating the lookahead
    obj_values, trajectories, actions = eval_agent(train_env, 
                                                    policy, 
                                                    problem_list=test_problem_list, 
                                                    num_steps=model_training_steps,
                                                    random_actions=(problemclass_train == "none"), 
                                                    )
    results[optimizer_name]['obj_values'] = obj_values
    results[optimizer_name]['trajectories'] = trajectories
    results[optimizer_name]['actions'] = actions

    # EVALUATE A RANDOM AGENT
    """
    optimizer_name = 'agent_random'
    results[optimizer_name] = {}
    train_env.train_mode = False # remove train mode, avoids calculating the lookahead
    obj_values, trajectories, actions = eval_agent(train_env, 
                                                    policy, 
                                                    problem_list=test_problem_list, 
                                                    num_steps=model_training_steps,
                                                    random_actions=True 
                                                    )
    results[optimizer_name]['obj_values'] = obj_values
    results[optimizer_name]['trajectories'] = trajectories
    results[optimizer_name]['actions'] = actions
    """

    params_dict = {
        'test_starting_points': test_starting_points,
        'threshold': threshold,
        'meshgrid': (X, Y, Z),

    }

    return results, params_dict

def agent_statistics(results, params_dict, do_plot=True):
    if do_plot:
        import matplotlib.pyplot as plt
    else :
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt


    test_starting_points = params_dict['test_starting_points']
    threshold = params_dict['threshold']
    X, Y, Z = params_dict['meshgrid']
    
    #### DISPLAY RESULTS ####

    # calculate a score matrix storing the mean objective value for each optimizer
    score_matrix = np.zeros((nb_test_points, len(results.keys())))
    for j, optimizer_name in enumerate(results.keys()):
        obj_values = results[optimizer_name]['obj_values']
        for i in range(nb_test_points):
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
    if do_plot:
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
    if do_plot:
        #analyze the actions taken by the agent
        optimizer_name = "agent"
        actions = results[optimizer_name]['actions']

        trajectories = results[optimizer_name]['trajectories']

        # if actions has 2 dims, expand it to 3 dims
        if len(actions.shape) == 2:
            actions = np.expand_dims(actions, axis=2)

        for actions_coeff_idx in range(actions.shape[-1]):
            beta = actions[:, :, actions_coeff_idx]
            # put all actions in a single array and plot the matrix 
            fig, ax = plt.subplots(1,2,figsize=(10, 10))
            ax[0].imshow(beta)
            ax[0].set_title('actions taken by the agent')
            ax[0].set_xlabel('step')
            ax[0].set_ylabel('starting point')


            # on the problem surface, plot agent actions for each starting point
            ax[1].contourf(X, Y, Z, 50, cmap="gray")
            ax[1].set_title('Agent actions for each starting point')
            ax[1].set_xlabel('x')
            ax[1].set_ylabel('y')
            for i in range(nb_test_points):
                ax[1].scatter(trajectories[i][:, 0], trajectories[i][:, 1], c=beta[i,:], s=1)
            plt.savefig("visualization/graphs/agent_actions.png")

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
    if do_plot:
        # select a starting point where the agent is the best optimizer
        idx_list = np.array(best_optimizer_list) == "agent"
        # if such a starting point exists, pick one at random
        if np.sum(idx_list) > 0:
            starting_point_idx = np.random.choice(np.where(idx_list)[0])
            
            # plot the trajectories of all optimizers for the selected starting point
            fig, ax = plt.subplots(1, 2, figsize=(10, 10))
            ax[0].contourf(X, Y, Z, 50, cmap="gray")
            ax[0].set_title('Trajectories for the selected starting point')
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('y')
            for i, optimizer_name in enumerate(optimizer_names):
                trajectories = results[optimizer_name]['trajectories']
                ax[0].plot(trajectories[starting_point_idx][:, 0], trajectories[starting_point_idx][:, 1], 'o-', label=optimizer_name)
            ax[0].legend()

            
            # plot the objective values for the selected starting point and optimizer
            ax[1].set_title('Objective values for the selected starting point')
            ax[1].set_xlabel('step')
            ax[1].set_ylabel('objective value')
            for i, optimizer_name in enumerate(optimizer_names):
                obj_values = results[optimizer_name]['obj_values']
                ax[1].plot(obj_values[starting_point_idx], label=optimizer_name)
            ax[1].legend()
            plt.savefig("visualization/graphs/objective_values_selected.png")

            plt.show()  
    # count the number of times each optimizer is the best
    best_optimizer_count = {}
    for optimizer_name in optimizer_names:
        best_optimizer_count[optimizer_name] = np.mean(np.array(best_optimizer_list) == optimizer_name)

    plt.close('all')
    return best_optimizer_count, score_matrix


def get_problem_name(problemclass):
    return problemclass.__name__ if isinstance(problemclass, type) else problemclass


if __name__ == "__main__":

    problemclass_train = GaussianHillsProblem
    problemclass_eval = GaussianHillsProblem

    problemclass_eval_name = problemclass_eval.__name__ if isinstance(problemclass_eval, type) else problemclass_eval
    problemclass_train_name = problemclass_train.__name__ if isinstance(problemclass_train, type) else problemclass_train

    filename = "visualization/result_dicts/"\
                + problemclass_train_name\
                + "_" + problemclass_eval_name\
                + config.policy.optimization_mode \
                + "_results.pkl"

    #if file already exists, load it
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            results, params_dict = pickle.load(f)

    #else, run the experiment and save the results
    else:
        results, params_dict = train_and_eval_agent(problemclass_train, problemclass_eval, agent_training_timesteps)
        with open(filename, 'wb') as f:
            pickle.dump((results, params_dict), f)

    best_optimizer_count, score_matrix = agent_statistics(results, params_dict, do_plot=True)
    print(best_optimizer_count)