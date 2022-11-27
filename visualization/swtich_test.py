import os, pickle
from problem import NoisyHillsProblem, GaussianHillsProblem, RosenbrockProblem\
    ,RastriginProblem, SquareProblemClass, AckleyProblem, NormProblem, \
        YNormProblem
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from eval_functions import eval_agent, eval_handcrafted_optimizer, eval_switcher_optimizer, first_index_below_threshold
import torch
from omegaconf import OmegaConf
from eval_functions import first_index_below_threshold
from environment import Environment
from stable_baselines3.common.env_checker import check_env
import stable_baselines3
import copy


config = OmegaConf.load('config.yaml')
xlim = 1
num_steps = 100
starting_point = np.array([0.5, 0.5])
problem = GaussianHillsProblem(x0=starting_point)
model = problem.model0
# meshgrid for plotting the problem surface
x = np.arange(-xlim, xlim, xlim / 100)
y = np.arange(-xlim, xlim, xlim / 100)
X, Y = np.meshgrid(x, y)
X, Y = torch.tensor(X), torch.tensor(Y)
Z = problem.function_def(X, Y)
Z = Z.detach().numpy()
optimizer_class_list = [torch.optim.SGD, torch.optim.Adam]
optimizer_class = optimizer_class_list[1]

# list of 0 and 1 of size num_steps, alternating every 10 steps
actions = [int(i%10 < 5) for i in range(num_steps)]
#actions = [1]*num_steps


trained_optimizers = dict.fromkeys(optimizer_class_list)
trained_optimizers_states = dict.fromkeys(optimizer_class_list)

for key, _ in trained_optimizers.items():
    # initialise the optimisers
    optimizer_init = key(model.parameters(), lr=config.model.lr)
    trained_optimizers[key] = optimizer_init
    trained_optimizers_states[key] = optimizer_init.state_dict()

use_state_dict = True
trajectory = []
o_v = []

for step in range(num_steps):
    for opt_class in optimizer_class_list:
        if opt_class != optimizer_class_list[actions[step]]:
            model_decoy = copy.deepcopy(model)
            
            # get optimizer or only the state dict?
            if use_state_dict:
                optimizer = opt_class(model_decoy.parameters(), lr=config.model.lr)
                optimizer.load_state_dict(trained_optimizers_states[opt_class])
            else :
                optimizer = trained_optimizers[opt_class]



            with torch.enable_grad():
                obj_value = problem.obj_function(model_decoy)
                optimizer.zero_grad()
                obj_value.backward()
                optimizer.step()
            # add the updated optimizer into list

            trained_optimizers[opt_class] = optimizer
            trained_optimizers_states[opt_class] = optimizer.state_dict()

        
    # get the optimizer to use
    if use_state_dict:
        optimizer = optimizer_class_list[actions[step]](model.parameters(), lr=config.model.lr)
        optimizer_state_dict = trained_optimizers_states[optimizer_class_list[actions[step]]]
        optimizer.load_state_dict(optimizer_state_dict)
    else : 
        optimizer = trained_optimizers[optimizer_class_list[actions[step]]]
    # or only the state dict?

        
    obj_value = problem.obj_function(model)
    o_v.append(obj_value.detach().numpy())
    trajectory.append(copy.deepcopy(model).x.detach().numpy())
    optimizer.zero_grad()
    obj_value.backward()
    optimizer.step()

    # print the state dict of Adam
    state = trained_optimizers_states[torch.optim.Adam]["state"]
    print(list(state.values()))
    state = trained_optimizers[torch.optim.Adam].state_dict()["state"]
    print(list(state.values()))

o_v = np.array(o_v)
trajectory = np.array(trajectory)

fig, ax = plt.subplots(1,2,figsize=(10, 10))
#plot learning curve
ax[0].plot(o_v)

ax[1].contourf(X, Y, Z, 50, cmap="gray")
ax[1].set_title('final objective value : {}'.format(o_v[-1]))
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
print(trajectory.shape)
ax[1].scatter(trajectory[:, 0], trajectory[:, 1], c=actions, s=1)
plt.show()

print("Final objective value: ", o_v[-1])