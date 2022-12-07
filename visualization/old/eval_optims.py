import sys
sys.path.append('..')
from problem import NoisyHillsProblem, GaussianHillsProblem, RosenbrockProblem\
    ,RastriginProblem, SquareProblemClass, AckleyProblem
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from eval_functions import eval_agent, eval_handcrafted_optimizer, eval_switcher_optimizer, first_index_below_threshold
import torch
from omegaconf import OmegaConf
from eval_functions import first_index_below_threshold

problemclass = GaussianHillsProblem
xlim = 2
nb_eval_points = 100
num_steps = 100
threshold = 0.001
config = OmegaConf.load('../config.yaml')

# generate list of random starting points
starting_points = np.random.uniform(-xlim, xlim, size=(nb_eval_points, 2))
#starting_points = np.arange(-0.5, -0.25, 0.01)

optimizer_class_list = [torch.optim.SGD,torch.optim.Adam]

problem_list = [problemclass(x0=xi) for xi in starting_points]

last_values = []

for optimizer in optimizer_class_list:
    loss_curves, trajectories = eval_handcrafted_optimizer(problem_list, optimizer, num_steps, config, do_init_weights=False)

    fibt = np.array([first_index_below_threshold(lc, threshold) for lc in loss_curves])
    
    # plot surface using starting points and fibt
    plt.scatter(starting_points[:,0], starting_points[:,1], c=loss_curves[:,-1], cmap='viridis')
    plt.colorbar()
    plt.show()

    print(loss_curves[:,-1])
    last_values.append(loss_curves[:,-1])

loss_curves = eval_switcher_optimizer(problem_list, 
                        optimizer_class_list, 
                        num_steps, 
                        config, 
                        switch_time=0.5, 
                        do_init_weights=False)

last_values.append(loss_curves[:,-1])

best_optimizer = np.argmin(np.array(last_values), axis=0)

optimizer_name_list = [optimizer.__name__ for optimizer in optimizer_class_list] + ['switcher']
# plot best optimizer for each starting point, with legend
for b in np.unique(best_optimizer):
    plt.scatter(starting_points[best_optimizer==b,0], starting_points[best_optimizer==b,1], label=optimizer_name_list[b])
plt.legend()
plt.show()
"""
plt.scatter(starting_points[:,0], starting_points[:,1], c=best_optimizer, cmap='viridis')
plt.colorbar()
plt.show()
"""







