import os, pickle
from problem import NoisyHillsProblem, GaussianHillsProblem, RosenbrockProblem\
    ,RastriginProblem, SquareProblemClass, AckleyProblem, NormProblem, \
        YNormProblem
from visualization.train_and_eval_agent import train_and_eval_agent, agent_statistics
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import wandb


config = OmegaConf.load('config.yaml')
num_agent_runs = config.model.num_agent_runs
model_training_steps = config.model.model_training_steps
agent_training_timesteps = num_agent_runs * model_training_steps


if __name__ == "__main__":

    problemclass_train_list = [GaussianHillsProblem]
    problemclass_eval_list = [GaussianHillsProblem]

    # every possible pair of problemclass 
    problemclass_pairs = [(train, eval) for train in problemclass_train_list for eval in problemclass_eval_list]

    #agent_performance = []

    nb_timesteps_list = [1000, 5000, 10000, 50000, 100000]
    nb_trials = 1
    
    #all_optimizer_results = []
    #all_score_matrices = []
    #agent_performance = np.zeros((nb_trials, len(nb_timesteps_list)))

    for (problemclass_train, problemclass_eval) in problemclass_pairs:

        for i, nb_timesteps in enumerate(nb_timesteps_list):
            #print(nb_timesteps)
            #aor = []
            #asm = []
            for trial in range(nb_trials):

                run = wandb.init(reinit=True, 
                                project="switching_optimizers", 
                                config={"problemclass_train": problemclass_train.__name__,
                                        "problemclass_eval": problemclass_eval.__name__,
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
            
            
            """
                print("agent performance: ", best_optimizer_count["agent"])
                #agent_performance[trial, i] = best_optimizer_count["agent"]
                #aor.append(best_optimizer_count)
                #asm.append(score_matrix)
                
                # save the results

                filename = "visualization/parameter_sensitivity_results/"\
                    #+ problemclass1.__name__\
                    #+ "_" + problemclass2.__name__\
                    #+ "_" + config.policy.optimization_mode 

                with open(filename + 'agent_performance.pkl', 'wb') as f:
                    pickle.dump(agent_performance, f)
                # save all_optimizer_results
                with open(filename + 'all_optimizer_results.pkl', 'wb') as f:
                    pickle.dump(all_optimizer_results, f)
                # save all_score_matrices
                with open(filename + 'all_score_matrices.pkl', 'wb') as f:
                    pickle.dump(all_score_matrices, f)

            all_optimizer_results.append(aor)
            all_score_matrices.append(asm)
                
            print("mean agent performance: ", np.mean(agent_performance[:, i]))
            """


    