from problem import NoisyHillsProblem, GaussianHillsProblem,\
     RastriginProblem, AckleyProblem, NormProblem
from visualization.train_and_eval_agent import train_and_eval_agent, agent_statistics, get_problem_name
from omegaconf import OmegaConf
import time
import wandb


config = OmegaConf.load('config.yaml')
num_agent_runs = config.model.num_agent_runs
model_training_steps = config.model.model_training_steps
agent_training_timesteps = num_agent_runs * model_training_steps
all_problems_class_list = [NoisyHillsProblem, GaussianHillsProblem, RastriginProblem, AckleyProblem, NormProblem]

if __name__ == "__main__":



    problemclass_train_list = [GaussianHillsProblem] #all_problems_class_list + ["none","all_except_eval"]
    problemclass_eval_list = [NoisyHillsProblem] #all_problems_class_list

    # every possible pair of problemclass 
    problemclass_pairs = [(train, eval) for train in problemclass_train_list for eval in problemclass_eval_list]

    nb_timesteps_list = [1, 100000]
    nb_trials = 5
    
    

    for (problemclass_train, problemclass_eval) in problemclass_pairs:


        for trial in range(nb_trials):        
                
            for i, nb_timesteps in enumerate(nb_timesteps_list):
                
                print("Trial: ", trial)
                time_start = time.time()

                print("problemclass_train: ", problemclass_train)
                print("problemclass_eval: ", problemclass_eval)


                run = wandb.init(reinit=True, 
                                project="switching_optimizers", 
                                group = "generalization_hard_with_all_betas",
                                config={"problemclass_train": get_problem_name(problemclass_train),
                                        "problemclass_eval": get_problem_name(problemclass_eval),
                                        "nb_timesteps": nb_timesteps, 
                                        "optimization_mode" : config.policy.optimization_mode, 
                                        "reward_system": config.environment.reward_system,
                                        "history_len": config.model.history_len,
                                        "lr": config.model.lr,
                                        "exploration_fraction": config.policy.exploration_fraction})
                with run:        

                    results, params_dict = train_and_eval_agent(problemclass_train, problemclass_eval, nb_timesteps)
                    best_optimizer_count, score_matrix = agent_statistics(results, params_dict, do_plot=False)

                    wandb.log({"agent_score": best_optimizer_count["agent"],
                                "all_optimizers_scores": best_optimizer_count,
                                "all_trajectories": results,
                               "score_matrix": score_matrix,})  

                time_end = time.time()
                print("time elapsed: ", time_end - time_start)




    