
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

num_episodes = 2

test_problem_list = [MLPProblemClass()]

# optimizer classes
optimizer_class_list = [torch.optim.SGD, torch.optim.RMSprop, torch.optim.Adam]
history_len = config.model.history_len


#exp_fac = [0.1,0.2,0.5,0.6, 0.1,0.2,0.5,0.6, 0.8, 0.9] only for testing, currently the exp factor is drawn from a distribution
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
    
    

def create_ensemble(ensemble_size):
    
    policy_array = np.array([])
    #to store the actions decided by individual agents - to get majority vote later
    actions_array = np.zeros((ensemble_size,num_episodes, model_training_steps))
    
    #to store rewards attained by individul agents - for reporting and comparison purposes
    single_rewards_array = np.zeros((ensemble_size,num_episodes, model_training_steps))
    
    #initialise training problem , same class random starting point
    x_start = np.random.randint(-5,5)
    y_start = np.random.randint(-5,5)
    train_problem_list = [MLPProblemClass() for _ in range(5)]
    
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
        agent_exp_factor = np.random.uniform(0,1)
        train_policy = stable_baselines3.DQN('MlpPolicy', train_env, exploration_fraction=agent_exp_factor,
                                       tensorboard_log='tb_logs/norm')
        
        train_policy.learn(total_timesteps=agent_training_timesteps)
        np.append(train_policy, train_policy)
        trained_actions, trained_rewards = eval_agent(test_env, train_policy, num_episodes=num_episodes, num_steps=model_training_steps)

        #print(trained_actions.shape)
        actions_array[i,:,:] = trained_actions
        single_rewards_array[i,:,:]  = trained_rewards
        print("HI")
        print(single_rewards_array)
        
        
    #now we need to combine the info from the different agents
    
    #majority voting
    ensemble_actions=[]
    #print(actions_array)
    
    ensemble_actions = []
    
    #array becomes an array of  timesteps x actions
    
    for i in range(model_training_steps):
        actions_array = actions_array.astype(int)
        voted_action = np.bincount(actions_array[1:,:,i].flatten()).argmax()
        
        ensemble_actions.append(voted_action)
    
    
    #simulate the actions chosen by ensemble on the test problem to get the rewards
    test_env.reset()
    ensemble_rewards = []
    for action in ensemble_actions:
        _, reward, _, _ = test_env.step(action)
        ensemble_rewards.append(reward)
    
    
    return ensemble_rewards, ensemble_actions, single_rewards_array

ensemble_sizes = [13]
#dictionary of performance

performance_dict = dict.fromkeys(ensemble_sizes)

#plots loss across time
for size in ensemble_sizes:
    print("size of", size)
    ensemble_rewards, ensemble_actions, single_rewards = create_ensemble(size)
#print(len(single_rewards[0]), len(ensemble_rewards))
    
    performance_dict[size] = [ensemble_rewards, ensemble_actions, single_rewards]
print(ensemble_rewards)
#picking a random agent to check performance
agent_chosen = np.random.randint(1, 13 + 1) 


#plotting rewards - incomplete
for performance_record in performance_dict.keys():
    #print("performance")
    ensemble_rewards = performance_dict[performance_record][0]
    plt.plot(ensemble_rewards, label='ensemble of '+str(size)+' agents rewards')


plt.plot(np.mean(single_rewards[2], axis = 0), label="single rewards average")
plt.legend()
plt.show()

#print(single_rewards[agent_chosen])  
#plot first past the post
    

    
        
        
    #for reward in 




