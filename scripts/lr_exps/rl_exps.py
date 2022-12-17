import copy

import numpy as np
import torch
import tqdm

import matplotlib.pyplot as plt 

# from ray import tune

import stable_baselines3
from stable_baselines3.common import vec_env, monitor


from rl_utils import lr_Environment, make_observation
from lr_utils import minimize_custom, norm_function, rosen_function

x_dim = 1 # dimension of the problem
objective_function = norm_function
x_0 = 0.0 #torch.tensor([0.0], requires_grad=True)
optimizer_class = torch.optim.SGD #torch.optim.Adam,
lr_values = np.logspace(-6, 0, 20)
n_steps = 40

env_dataset = [norm_function for _ in range(90)]
norm_env = lr_Environment(env_dataset, num_steps=n_steps, history_len=25)

norm_policy = stable_baselines3.PPO('MlpPolicy', 
                                    norm_env, 
                                    n_steps=2, 
                                    verbose=0,
                                    tensorboard_log='tb_logs/norm')


total_timesteps = 20000 #20*40*90

#quadratic_policy.learn(total_timesteps=20 * quadratic_env.envs[0].num_steps * len(norm_dataset))
norm_policy.learn(total_timesteps=total_timesteps)




def run_policy(n_steps):
  actions = []
  obs = norm_env.reset()
  
  for step in range(n_steps):
    action, _ = norm_policy.predict(obs, deterministic=True)
    actions.append(lr_values[action])
    print("Step {}".format(step + 1))
    print("Action: ", lr_values[action])
    obs, reward, done, info = norm_env.step(action)
    #print('obs=', obs, 'reward=', reward, 'done=', done)
    print('reward=', reward)
    #norm_env.render(mode='console')
    if done or step == n_steps - 1:
      # Note that the VecEnv resets automatically
      # when a done signal is encountered
      print("Goal reached!", "reward=", reward)
      return actions

run_list = np.array([run_policy(n_steps) for _ in range(10)])

plt.plot(range(n_steps),np.mean(run_list, axis=0))
plt.show()