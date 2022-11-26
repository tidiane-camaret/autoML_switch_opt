
import os

from problem import *
import torch
from environment import *
import stable_baselines3
from train_agent import eval_agent
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from omegaconf import OmegaConf

config = OmegaConf.load('config.yaml')
# number of problems to train on
num_problems = config.model.num_problems

# number of agent training episodes
num_agent_runs = config.model.num_agent_runs
# number of steps in each episode
model_training_steps = config.model.model_training_steps

agent_training_timesteps = num_agent_runs * model_training_steps
# define the problem list
x_start_test = np.random.randint(-5,5)
y_start_test = np.random.randint(-5,5)

num_episodes = 10

test_problem_list = [Beale(x_start_test, y_start_test)]

# optimizer classes
optimizer_class_list = [torch.optim.SGD, torch.optim.RMSprop, torch.optim.Adam]
history_len = config.model.history_len

# number of agents in ensemble
ensemble_size = 3
exp_fac = [0.1,0.2,0.5,0.6, 0.1,0.2,0.5,0.6, 0.8, 0.9]
# define the environment based on the problem list


test_env = Environment(config=config,
                        problem_list=test_problem_list,
                        num_steps=model_training_steps,
                        history_len=history_len,
                        optimizer_class_list=optimizer_class_list,
                        do_init_weights=False
                        )

# sanity check for the environment

check_env(test_env, warn=True)

# define the agent

    
    

def create_ensemble():
    
    policy_array = np.array([])
    actions_array, rewards_array,single_rewards  = np.zeros((num_episodes, model_training_steps)), np.zeros((num_episodes, model_training_steps)), np.zeros((num_episodes, model_training_steps))
    
    
    x_start = np.random.randint(-5,5)
    y_start = np.random.randint(-5,5)
    train_problem_list = [Beale(x_start, y_start) for _ in range(5)]
    
    train_env = Environment(config=config,
                            problem_list=train_problem_list,
                            num_steps=model_training_steps,
                            history_len=history_len,
                            optimizer_class_list=optimizer_class_list,
                            do_init_weights=False
                            )
    check_env(train_env, warn=True)
    for i in range(ensemble_size):
        
        #create a training env for each agent
        
        #each agent trains on different problem set
        
        #change exploration factor for high variaty between agents
        train_policy = stable_baselines3.DQN('MlpPolicy', train_env, verbose=0, exploration_fraction=exp_fac[i],
                                       tensorboard_log='tb_logs/norm')
        
        train_policy.learn(total_timesteps=agent_training_timesteps)
        
        trained_actions, trained_rewards = eval_agent(test_env, train_policy, num_episodes=num_episodes, num_steps=model_training_steps)

        #add the actions
        #print(trained_actions, "tt")
        actions_array = np.concatenate((actions_array,trained_actions))
        single_rewards = np.concatenate((single_rewards,trained_rewards)) 
        #single_rewards.append([trained_rewards])
        
    #now we need to combine the info from the different agents
    
    #majority voting
    ensemble_actions=[]
    print(actions_array)
    
    #array becomes an array of  timesteps x actions
    actions_array = np.array(actions_array).transpose()
    
    #print(np.array(single_rewards[1:]), "RE SIG")
    #temporary, optimise later
    for time_step in actions_array:
        count = np.zeros(len(optimizer_class_list))
        for i in range(len(time_step)):
            action_chosen = time_step[i]
            #index = optimizer_class_list.index(action_chosen)
            count[int(action_chosen)] += 1
        ensemble_actions.append( np.argmax(count))
    print(ensemble_actions)
    #simulate the actionns chosen on the same problem
    test_env.reset()
    ensemble_rewards = []
    for action in ensemble_actions:
        obs, reward, _, _ = test_env.step(action)
        ensemble_rewards.append(reward)
    
    
    return ensemble_rewards, ensemble_actions, single_rewards


ensemble_rewards, ensemble_actions, single_rewards = create_ensemble()
print(len(single_rewards[0]), len(ensemble_rewards))
plt.plot(ensemble_rewards, label='ensemble rewards')
plt.plot(np.mean(single_rewards, axis = 0), label="single rewards average")

# plt.fill_between(np.arange(len(single_rewards[0])), np.mean(single_rewards, axis=0) - np.std(single_rewards, axis=0),
#                  np.mean(single_rewards, axis=0) + np.std(single_rewards, axis=0), alpha=0.2)

plt.legend()
plt.show()




