from main import train_and_eval_agent
import wandb
from problem import NoisyHillsProblem, GaussianHillsProblem, ImageDatasetProblemClass, AckleyProblem, RastriginProblem, NormProblem
import numpy as np
import torch
import torchvision.datasets
import random
### parameters specific to math problems
math_problem_train_class = GaussianHillsProblem
math_problem_eval_class = GaussianHillsProblem
xlim = 2
nb_train_points = 1000
nb_test_points = 500


sweep_config = {
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "agent_score"
    },
    "parameters": {
                    "problem_train": {
                        "values": ["All"]
                    },
                    "problem_test": {
                        "values": ["Gaussian", "Noisy", "Ackley", "Rastrigin", "Norm"]
                    },

                    "exploration_fraction": {
                        "values": [0.25]
                        },
                    "lr": {
                        "values": [0.01]
                        },
                    "history_len": {
                        "values": [15]
                        },
                    "nb_timesteps": {
                        "values": [80000]
                        #"min": 100,
                        #"max": 100000,
                        },
                    "reward_system": {
                        "values": ["inverse"]
                        },
                    "optimization_mode": {
                        "values": ["hard"]},

                    }
}




sweep_id = wandb.sweep(sweep_config, project="switching_optimizers")
#sweep_id = "switching_optimizers/qmr074oo"

def sweep_function():

    run = wandb.init()
    agent_training_timesteps = wandb.config.nb_timesteps
    exploration_fraction = wandb.config.exploration_fraction
    lr = wandb.config.lr
    history_len = wandb.config.history_len
    reward_system = wandb.config.reward_system
    optimization_mode = wandb.config.optimization_mode


    if wandb.config.problem_train == 'MNIST':
        nb_test_points = 100
        binary_classes = [[2,3], [4,5], [6,7], [8,9]]
        train_problem_list = [ImageDatasetProblemClass(classes = bc, dataset_class=torchvision.datasets.MNIST) for bc in binary_classes]
        test_problem_list = [ImageDatasetProblemClass(classes = [0,1], dataset_class=torchvision.datasets.MNIST) for _ in range(nb_test_points)]
        threshold = 0.05

    elif wandb.config.problem_train == 'All':
        xlim = 2
        nb_test_points = 500
        train_problem_list = [random.choice([GaussianHillsProblem, NoisyHillsProblem, AckleyProblem, RastriginProblem, NormProblem])(x0=np.random.uniform(-xlim, xlim, size=(2))) 
                            for _ in range(nb_train_points)]

    else :
        xlim = 2
        nb_test_points = 500
        if wandb.config.problem_train == 'Gaussian':
            math_problem_train_class = GaussianHillsProblem
        elif wandb.config.problem_train == 'Noisy':
            math_problem_train_class = NoisyHillsProblem
        elif wandb.config.problem_train == 'Ackley':
            math_problem_train_class = AckleyProblem
        elif wandb.config.problem_train == 'Rastrigin':
            math_problem_train_class = RastriginProblem
        elif wandb.config.problem_train == 'Norm':
            math_problem_train_class = NormProblem

        train_problem_list = [math_problem_train_class(x0=np.random.uniform(-xlim, xlim, size=(2))) 
                            for _ in range(nb_train_points)]

        

        if wandb.config.problem_test == 'Gaussian':
            math_problem_eval_class = GaussianHillsProblem
        elif wandb.config.problem_test == 'Noisy':
            math_problem_eval_class = NoisyHillsProblem
        elif wandb.config.problem_test == 'Ackley':
            math_problem_eval_class = AckleyProblem
        elif wandb.config.problem_test == 'Rastrigin':
            math_problem_eval_class = RastriginProblem
        elif wandb.config.problem_test == 'Norm':
            math_problem_eval_class = NormProblem




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


    optimizers_scores, optimizers_trajectories = train_and_eval_agent(train_problem_list=train_problem_list,
                        test_problem_list=test_problem_list,
                        agent_training_timesteps=agent_training_timesteps,
                        exploration_fraction = exploration_fraction,
                        history_len=history_len,
                        optimization_mode=optimization_mode,
                        lr=lr,
                        reward_system=reward_system,
                        threshold=threshold,
                        optimizer_class_list=optimizer_class_list,
                        do_plot=False)

    wandb.log({"optimizers_scores":optimizers_scores,
                "agent_score": optimizers_scores["agent"],

            })

wandb.agent(sweep_id, function=sweep_function)