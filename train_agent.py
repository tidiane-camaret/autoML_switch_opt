from environment import Environment
from problem import MLP_problem
import torch 
from stable_baselines3.common.env_checker import check_env
import stable_baselines3

# define the problem list
problem_list = [MLP_problem for _ in range(10)]


# define the environment based on the problem list
env = Environment(problem_list = problem_list,
                 model =  MLP_problem["model0"],
                 num_steps = 100, 
                 history_len = 10, 
                 objective_function = MLP_problem["obj_function"],
                 optimizer_class_list=[torch.optim.SGD, torch.optim.Adam]
                 )

# sanity check for the environment
check_env(env, warn=True)

# define the agent
policy = stable_baselines3.PPO('MlpPolicy', env, n_steps=2, verbose=0,
                                         tensorboard_log='tb_logs/norm')

# train the agent
total_timesteps = 1000
policy.learn(total_timesteps=total_timesteps)



# test the agent
# Test the trained agent
actions = []
obs = env.reset()
n_steps = 20
for step in range(n_steps):
  action, _ = policy.predict(obs, deterministic=True)
  actions.append(action)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  #print('obs=', obs, 'reward=', reward, 'done=', done)
  print('reward=', reward)
  #norm_env.render(mode='console')
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)
    break