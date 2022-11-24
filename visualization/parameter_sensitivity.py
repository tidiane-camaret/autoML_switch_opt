import os, pickle
from problem import NoisyHillsProblem, GaussianHillsProblem, RosenbrockProblem\
    ,RastriginProblem, SquareProblemClass, AckleyProblem, NormProblem, \
        YNormProblem
from train_and_eval_agent import train_and_eval_agent, agent_statistics
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt

config = OmegaConf.load('config.yaml')
num_agent_runs = config.model.num_agent_runs
model_training_steps = config.model.model_training_steps
agent_training_timesteps = num_agent_runs * model_training_steps


if __name__ == "__main__":

    problemclass1 = GaussianHillsProblem
    problemclass2 = problemclass1

    agent_performance = []

    nb_timesteps_list = [1, 10, 100, 1000, 5000, 10000, 50000, 100000, 500000]
    nb_trials = 10
    all_optimizer_results = []

    agent_performance = np.zeros((nb_trials, len(nb_timesteps_list)))

    for i, nb_timesteps in enumerate(nb_timesteps_list):
        print(nb_timesteps)
        aor = []
        for trial in range(nb_trials):
            results, params_dict = train_and_eval_agent(problemclass1, problemclass2, nb_timesteps)
            best_optimizer_count = agent_statistics(results, params_dict, do_plot=False)

            agent_performance[trial, i] = best_optimizer_count["agent"]
            aor.append(best_optimizer_count)

            # save the results
            with open('visualization/agent_performance.pkl', 'wb') as f:
                pickle.dump(agent_performance, f)
            # save all_optimizer_results
            with open('visualization/all_optimizer_results.pkl', 'wb') as f:
                pickle.dump(all_optimizer_results, f)

        
        all_optimizer_results.append(aor)
            
        print("mean agent performance: ", np.mean(agent_performance[:, i]))

    plt.plot(nb_timesteps_list, agent_performance)

    