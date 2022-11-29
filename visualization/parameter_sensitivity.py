import os, pickle
from problem import NoisyHillsProblem, GaussianHillsProblem, RosenbrockProblem\
    ,RastriginProblem, SquareProblemClass, AckleyProblem, NormProblem, \
        YNormProblem
from visualization.train_and_eval_agent import train_and_eval_agent, agent_statistics, get_problem_name
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import wandb


config = OmegaConf.load('config.yaml')
num_agent_runs = config.model.num_agent_runs
model_training_steps = config.model.model_training_steps
agent_training_timesteps = num_agent_runs * model_training_steps


if __name__ == "__main__":

    problemclass_train_list = ["all_except_eval"]
    problemclass_eval_list = [GaussianHillsProblem, NoisyHillsProblem, RastriginProblem, AckleyProblem, NormProblem]

    # every possible pair of problemclass 
    problemclass_pairs = [(train, eval) for train in problemclass_train_list for eval in problemclass_eval_list]

    nb_timesteps_list = [100000]
    nb_trials = 5
    

    for (problemclass_train, problemclass_eval) in problemclass_pairs:

        for i, nb_timesteps in enumerate(nb_timesteps_list):
            #print(nb_timesteps)
            #aor = []
            #asm = []
            for trial in range(nb_trials):

                run = wandb.init(reinit=True, 
                                project="switching_optimizers", 
                                config={"problemclass_train": get_problem_name(problemclass_train),
                                        "problemclass_eval": get_problem_name(problemclass_eval),
                                        "nb_timesteps": nb_timesteps, 
                                        "optimization_mode" : config.policy.optimization_mode, 
                                        "reward_system": config.environment.reward_system,
                                        "history_len": config.model.history_len,
                                        "lr": config.model.lr,})
                with run:        

                    results, params_dict = train_and_eval_agent(problemclass_train, problemclass_eval, nb_timesteps)
                    best_optimizer_count, score_matrix = agent_statistics(results, params_dict, do_plot=False)

                    wandb.log({"agent_count": best_optimizer_count["agent"],
                                "all_optimizer_count": best_optimizer_count,
                                "all_trajectories": results,
                               "score_matrix": score_matrix,})    



    