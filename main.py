import random
import os, glob
import wandb
from problem import NoisyHillsProblem, GaussianHillsProblem, RosenbrockProblem\
    ,RastriginProblem, SquareProblemClass, AckleyProblem, NormProblem, \
        YNormProblem, MNISTProblemClass, ImageDatasetProblemClass
import torch
from environment import Environment
from eval_functions import eval_agent, eval_handcrafted_optimizer
import stable_baselines3
from eval_functions import eval_agent
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from omegaconf import OmegaConf
from collections import Counter
from stable_baselines3.common import vec_env, monitor
from stable_baselines3.common.env_util import make_vec_env
import torchvision.datasets

num_cpu = 3


config = OmegaConf.load('config.yaml')
tb_log_dir = 'tb_logs/'+config.problem.train+' '+config.policy.optimization_mode
# number of agent training episodes
num_agent_runs = config.model.num_agent_runs
# number of steps in each episode
model_training_steps = config.model.model_training_steps

agent_training_timesteps = num_agent_runs * model_training_steps
history_len = config.model.history_len
exploration_fraction = config.policy.exploration_fraction
lr = config.model.lr
# define the problem list


### parameters specific to math problems



reward_system = config.environment.reward_system
optimization_mode = config.policy.optimization_mode

if config.problem.train == 'MNIST':
    nb_test_points = 100
    binary_classes = [[2,3], [4,5], [6,7], [8,9]]
    train_problem_list = [ImageDatasetProblemClass(classes = bc, dataset_class=torchvision.datasets.MNIST) for bc in binary_classes]
    test_problem_list = [ImageDatasetProblemClass(classes = [0,1], dataset_class=torchvision.datasets.MNIST) for _ in range(nb_test_points)]
    threshold = 0.05


else :
    xlim = 2
    nb_test_points = 500
    if config.problem.train == 'Gaussian':
        math_problem_train_class = GaussianHillsProblem
    elif config.problem.train == 'Noisy':
        math_problem_train_class = NoisyHillsProblem
    elif config.problem.train == 'Ackley':
        math_problem_train_class = AckleyProblem
    elif config.problem.train == 'Rastrigin':
        math_problem_train_class = RastriginProblem
    elif config.problem.train == 'Norm':
        math_problem_train_class = NormProblem
    

    if config.problem.test == 'Gaussian':
        math_problem_eval_class = GaussianHillsProblem
    elif config.problem.test == 'Noisy':
        math_problem_eval_class = NoisyHillsProblem
    elif config.problem.test == 'Ackley':
        math_problem_eval_class = AckleyProblem
    elif config.problem.test == 'Rastrigin':
        math_problem_eval_class = RastriginProblem
    elif config.problem.test == 'Norm':
        math_problem_eval_class = NormProblem


    
    train_problem_list = [math_problem_train_class(x0=np.random.uniform(-xlim, xlim, size=(2))) 
                        for _ in range(num_agent_runs)]
    test_problem_list = [math_problem_eval_class(x0=np.random.uniform(-xlim, xlim, size=(2))) 
                        for _ in range(nb_test_points)]
    
    # meshgrid for plotting the problem surface
    x = np.arange(-xlim, xlim, xlim / 100)
    y = np.arange(-xlim, xlim, xlim / 100)
    X, Y = np.meshgrid(x, y)
    X, Y = torch.tensor(X), torch.tensor(Y)
    Z = math_problem_eval_class().function_def(X, Y)
    Z = Z.detach().numpy()

    # calculate minimum of the problem surface
    # and determine the threshold for the reward function
    function_min = np.min(Z)
    #print('test function minimum: ', function_min)
    threshold = function_min + 0.001


# optimizer classes
optimizer_class_list = [torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop]


# define the environment based on the problem list


def train_and_eval_agent(train_problem_list=train_problem_list,
                        test_problem_list=test_problem_list,
                        agent_training_timesteps=agent_training_timesteps,
                        exploration_fraction = exploration_fraction,
                        history_len=history_len,
                        optimization_mode=optimization_mode,
                        lr=lr,
                        reward_system=reward_system,
                        threshold=threshold,
                        optimizer_class_list=optimizer_class_list,
                        do_plot=True):

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

    # sanity check for the environment
    check_env(train_env, warn=True)

    # vectorize the environment for parallelization
    vec_train_env = make_vec_env(lambda: train_env, n_envs=num_cpu)


    # define the agent
    if config.policy.model == 'PPO' or config.policy.optimization_mode == "soft":
        policy = stable_baselines3.PPO('MlpPolicy',
                                    vec_train_env, 
                                    verbose=0,
                                    tensorboard_log=tb_log_dir,
                                    device='cpu')

    elif config.policy.model == 'DQN':
        policy = stable_baselines3.DQN('MlpPolicy',
                                    vec_train_env, 
                                    #buffer_size=100_000, 
                                    verbose=0,
                                    exploration_fraction=exploration_fraction,
                                    tensorboard_log=tb_log_dir,
                                    device='cpu')




    optimizers_trajectories = {}


    # train the agent
    train_problem_name = train_problem_list[0].__class__.__name__
    saved_agent_path = 'models/trained_agent_' + train_problem_name + '_' + config.environment.reward_system

    # if an agent is already trained, load it
    if glob.glob(saved_agent_path+'*'):
        print('loading trained agent ...')
        policy= stable_baselines3.DQN.load(saved_agent_path, env=vec_train_env)
        
    
    # otherwise train a new agent
    else:
        print('training agent ...')
        policy.learn(total_timesteps=agent_training_timesteps, 
                    progress_bar=True,
                    #eval_freq=1000, 
                    #eval_log_path=tb_log_dir
                    )
        policy.save(saved_agent_path)

    # evaluate the agent
    train_env.train_mode = False # remove train mode, avoids calculating the lookahead
    print("evaluating the agent ...")
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
    print("evaluating a random agent ...")
    optimizers_trajectories['random_agent'] = {}

    obj_values, trajectories, actions = eval_agent(train_env, 
                                                    policy, 
                                                    problem_list=test_problem_list, 
                                                    num_steps=model_training_steps,
                                                    random_actions=True,
                                                    )
    optimizers_trajectories['random_agent']['obj_values'] = obj_values
    optimizers_trajectories['random_agent']['trajectories'] = trajectories
    optimizers_trajectories['random_agent']['actions'] = actions


    # evaluate the handcrafted optimizers
    for optimizer_class in optimizer_class_list:
        optimizer_name = optimizer_class.__name__
        print("evaluating ", optimizer_name)
        optimizers_trajectories[optimizer_name] = {}
        obj_values, trajectories = eval_handcrafted_optimizer(test_problem_list, 
                                                                optimizer_class, 
                                                                model_training_steps, 
                                                                config, 
                                                                do_init_weights=False,
                                                                lr = test_problem_list[0].tuned_lrs[optimizer_class],)

        optimizers_trajectories[optimizer_name]['obj_values'] = obj_values
        optimizers_trajectories[optimizer_name]['trajectories'] = trajectories






    # calculate a score matrix storing the aera under the curve for each optimizer
    # and each starting point
    score_matrix = np.zeros((nb_test_points, len(optimizers_trajectories.keys())))
    for j, optimizer_name in enumerate(optimizers_trajectories.keys()):
        obj_values = optimizers_trajectories[optimizer_name]['obj_values']
        for i in range(nb_test_points):
            #score_matrix[i, j] = first_index_below_threshold(obj_values[i], threshold)
            score_matrix[i, j] = np.mean(obj_values[i][:])



    # list of best optimizers for each starting point. 
    # if the agent is in a tie with a handcrafted optimizer, the agent wins
    best_optimizer_list = []
    for i in range(nb_test_points):
        best_score = np.inf
        best_optimizer = "agent"
        for j, optimizer_name in enumerate(optimizers_trajectories.keys()):
            if score_matrix[i, j] < best_score:
                best_score = score_matrix[i, j]
                best_optimizer = optimizer_name
        best_optimizer_list.append(best_optimizer)


    optimizers_scores = {}
    for optimizer_name in optimizers_trajectories.keys():
        optimizers_scores[optimizer_name] = np.mean(np.array(best_optimizer_list) == optimizer_name)



    print("optimizers scores : ", optimizers_scores)

    if do_plot:

        # plot mean objective value for each optimizer on the same plot
        plt.figure(figsize=(10, 6))
        for optimizer_name, optimizer_trajectories in optimizers_trajectories.items():
            obj_values = optimizer_trajectories['obj_values']
            plt.plot(np.mean(obj_values, axis=0), label=optimizer_name)
            print(optimizer_name, " mean obj value : ", np.mean(obj_values, axis=0)[-1])
        plt.legend()
        plt.show()

        # plot the histogram of the scores for each optimizer
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for j, optimizer_name in enumerate(optimizers_trajectories.keys()):
            ax.hist(score_matrix[:, j], bins=100, label=optimizer_name, alpha=0.5)
        ax.legend()
        ax.set_xlabel('mean objective value')
        ax.set_ylabel('number of test points')
        ax.set_title('Histogram of the mean objective value for each optimizer')


        # get indices where the agent is the best optimizer
        plotted_starting_points = [i for i, x in enumerate(best_optimizer_list) if x == "agent"]
        if len(plotted_starting_points) < 10:
            plotted_starting_points.extend(random.sample(range(nb_test_points), 10-len(plotted_starting_points)))
        else:
            plotted_starting_points = plotted_starting_points[:10]

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


        #analyze the actions taken by the agent
        actions = optimizers_trajectories["agent"]['actions']

        # if actions has 2 dims, expand it to 3 dims
        if len(actions.shape) == 2:
            actions = np.expand_dims(actions, axis=2)

        for actions_coeff_idx in range(actions.shape[-1]):
            beta = actions[:, :, actions_coeff_idx]
            # put all actions in a single array and plot the matrix 
            fig, ax = plt.subplots(1,figsize=(10, 10))
            ax.imshow(beta)
            ax.set_title('actions taken by the agent')
            ax.set_xlabel('step')
            ax.set_ylabel('starting point')
            plt.show()


    return optimizers_scores, optimizers_trajectories

if __name__ == '__main__':
    run = wandb.init(reinit=True, 
                                    project="switching_optimizers", 
                                    group = "main",
                                    config={"problem": config.problem,
                                            "nb_timesteps": agent_training_timesteps, 
                                            "optimization_mode" : config.policy.optimization_mode, 
                                            "reward_system": config.environment.reward_system,
                                            "history_len": config.model.history_len,
                                            "lr": config.model.lr,
                                            "exploration_fraction": config.policy.exploration_fraction})


    optimizers_scores, optimizers_trajectories = train_and_eval_agent()

    wandb.log({"optimizers_scores":optimizers_scores,
            "agent_actions":optimizers_trajectories['agent']['actions'],
            #"optimizers_trajectories":optimizers_trajectories,
            })
    #wandb.log({
    #        #"agent_actions":optimizers_trajectories['agent']['actions'],
    #        "optimizers_trajectories":optimizers_trajectories,
    #        })