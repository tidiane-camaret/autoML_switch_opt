from problem import MLPProblemClass, RosenbrockProblemClass, SquareProblemClass
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from environment import eval_handcrafted_optimizer, eval_switcher_optimizer
import torch
from problem_evaluation import first_index_below_threshold

config = OmegaConf.load('config.yaml')
num_problems = 10
num_steps = 100
optimizer_class_list = [torch.optim.SGD,torch.optim.Adam]
threshold = 0.05
switch_time = 0.5
starting_points = np.arange(-1, 1, 0.01)
problem_list = [
    SquareProblemClass(
                        x0=x0 
                )
                        for x0 in starting_points] 

# plot the function of the first problem
x = np.linspace(0.5, 1, 100)
y = [problem_list[0].function_def(xi) for xi in x]

#plt.plot(x, y, alpha=0.5)   

fibts = []
for optimizer in optimizer_class_list:
    loss_curves, trajectories = eval_handcrafted_optimizer(problem_list, optimizer, num_steps, config, do_init_weights=False)
    loss_curves = - loss_curves

    fibts.append(
        np.apply_along_axis(
            first_index_below_threshold, 1, loss_curves, threshold
            )
                )

    #plot the trajectory of the first problem
    #plt.plot(trajectories[0], loss_curves[0], label=optimizer.__name__, marker="o", markersize=5)
    plt.plot(loss_curves[0], label=optimizer.__name__)

loss_curves = - 1 * eval_switcher_optimizer(problem_list, 
                                            optimizer_class_list, 
                                            num_steps, 
                                            config, 
                                            switch_time, 
                                            do_init_weights=False)

fibts.append(
    np.apply_along_axis(
        first_index_below_threshold, 1, loss_curves, threshold
        )
            )

plt.legend()
plt.show()

fibts = np.array(fibts)
print(fibts)
print(np.argmin(fibts, axis=0))
plt.plot(starting_points, fibts[0], label=optimizer_class_list[0].__name__)
plt.plot(starting_points, fibts[1], label=optimizer_class_list[1].__name__)
plt.plot(starting_points, fibts[2], label="switcher")

plt.title("First time step below " + str(threshold))
plt.legend()
plt.show()