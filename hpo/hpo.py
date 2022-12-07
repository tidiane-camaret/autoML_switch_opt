from main import train_and_eval_agent
import wandb
from problem import NoisyHillsProblem, GaussianHillsProblem
import numpy as np
import torch
### parameters specific to math problems
math_problem_train_class = GaussianHillsProblem
math_problem_eval_class = GaussianHillsProblem
xlim = 2
nb_train_points = 1000
nb_test_points = 500


train_problem_list = [math_problem_train_class(x0=np.random.uniform(-xlim, xlim, size=(2))) 
                    for _ in range(nb_train_points)]
test_problem_list = [math_problem_eval_class(x0=np.random.uniform(-xlim, xlim, size=(2))) 
                    for _ in range(nb_test_points)]




sweep_config = {
    "method": "random",
    "metric": {
        "goal": "maximize",
        "name": "agent_score"
    },
    "parameters": {"exploration_fraction": {
                        "values": [0.25]
                        },
                    "lr": {
                        "values": [0.01]
                        },
                    "history_len": {
                        "values": [15]#[1, 15, 25, 35, 50 ]
                        },
                    "nb_timesteps": {
                        #"values": [100, 1000, 10000, 20000, 50000, 100000]
                        "min": 100,
                        "max": 100000,
                        },
                    "reward_system": {
                        "values": ["lookahead","threshold", "inverse", "opposite"]
                        },
                    "optimization_mode": {
                        "values": ["hard"]},

                    }
}




#sweep_id = wandb.sweep(sweep_config, project="switching_optimizers")
sweep_id = "switching_optimizers/qmr074oo"

def sweep_function():
    run = wandb.init()
    agent_training_timesteps = wandb.config.nb_timesteps
    exploration_fraction = wandb.config.exploration_fraction
    lr = wandb.config.lr
    history_len = wandb.config.history_len
    reward_system = wandb.config.reward_system
    optimization_mode = wandb.config.optimization_mode

    

    optimizers_scores, optimizers_trajectories = train_and_eval_agent(train_problem_list=train_problem_list,
                        test_problem_list=test_problem_list,
                        agent_training_timesteps=agent_training_timesteps,
                        exploration_fraction = exploration_fraction,
                        history_len=history_len,
                        optimization_mode=optimization_mode,
                        lr=lr,
                        reward_system=reward_system,
                        do_plot=False)

    wandb.log({"optimizers_scores":optimizers_scores,
                "agent_score": optimizers_scores["agent"],

            })

wandb.agent(sweep_id, function=sweep_function)