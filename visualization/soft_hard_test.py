import os, pickle
from problem import NoisyHillsProblem, GaussianHillsProblem, RosenbrockProblem\
    ,RastriginProblem, SquareProblemClass, AckleyProblem, NormProblem, \
        YNormProblem
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from eval_functions import eval_agent, eval_handcrafted_optimizer
import torch
from omegaconf import OmegaConf
from eval_functions import first_index_below_threshold
from environment import Environment
from stable_baselines3.common.env_checker import check_env
import stable_baselines3
import copy
from modifedAdam import ModifiedAdam

config = OmegaConf.load('config.yaml')
xlim = 1
num_steps = 100
starting_point = np.random.uniform(-xlim, xlim, size=(2))
problem = GaussianHillsProblem(x0=starting_point)
model = problem.model0
# meshgrid for plotting the problem surface
x = np.arange(-xlim, xlim, xlim / 100)
y = np.arange(-xlim, xlim, xlim / 100)
X, Y = np.meshgrid(x, y)
X, Y = torch.tensor(X), torch.tensor(Y)
Z = problem.function_def(X, Y)
Z = Z.detach().numpy()
optimizer_class_list = [torch.optim.SGD, torch.optim.Adam, ModifiedAdam]


trajectory = []
obj_values = []

for optimizer_class in optimizer_class_list:
    model_copy = copy.deepcopy(model)
    optimizer = optimizer_class(model_copy.parameters(), lr=config.model.lr)
    t = []
    o_v = []
    for step in range(num_steps):
        if optimizer_class == ModifiedAdam:
            beta1, beta2 = np.random.uniform([0.01, 0.998], [0.999, 0.999])
            for param in optimizer.param_groups:
                param['betas'] = (beta1, beta2, 0, 0)  

        
            
        obj_value = problem.obj_function(model_copy)
        
        t.append(copy.deepcopy(model_copy).x.detach().numpy())
        o_v.append(obj_value.detach().numpy())
        optimizer.zero_grad()
        obj_value.backward()
        optimizer.step()
    
    trajectory.append(t)
    obj_values.append(o_v)

trajectory = np.array(trajectory)
obj_values = np.array(obj_values)

print(trajectory.shape)
print(obj_values.shape)

fig, ax = plt.subplots(1,2,figsize=(10, 10))
#plot learning curve
for i in range(len(optimizer_class_list)):
    ax[0].plot(obj_values[i], label=optimizer_class_list[i].__name__)
ax[0].set_xlabel('step')
ax[0].set_ylabel('objective value')
ax[0].legend()


ax[1].contourf(X, Y, Z, 50, cmap="gray")
ax[1].set_title('final objective value : {}'.format(o_v[-1]))
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
for i in range(len(optimizer_class_list)):
    ax[1].plot(trajectory[i][:,0], trajectory[i][:,1], label=optimizer_class_list[i].__name__)
ax[1].legend()

plt.show()

print("Final objective values: ", obj_values[:,-1])