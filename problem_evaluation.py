from problem import MLPProblemClass
from environment import eval_handcrafted_optimizer, eval_switcher_optimizer
import torch
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt

def first_index_below_threshold(array, threshold):
    """Return the first index of an array below a threshold. if none, return last index."""
    for i, x in enumerate(array):
        if x < threshold:
            return i
    return len(array)-1



config = OmegaConf.load('config.yaml')
num_problems = 30
num_steps = 100
threshold = 0.6

optimizer = torch.optim.Adam
optimizer_class_list = [torch.optim.Adam,torch.optim.RMSprop]

switch_times = np.arange(0, 1.2, 0.2)


problem_list = [MLPProblemClass(
                        num_vars=8,
                        num_gaussians=30,
                )

                        for _ in range(num_problems)]

# plot mean loss curve for each optimizer
for optimizer in optimizer_class_list:
    loss_curves = - 1 * eval_handcrafted_optimizer(problem_list, optimizer, num_steps, config, do_init_weights=False)
    mean_loss_curve = np.mean(loss_curves, axis=0)
    plt.plot(mean_loss_curve, label=optimizer.__name__)

plt.legend()
plt.show()
"""
# plot the average curve and std
plt.plot(np.mean(loss_curves, axis=0))
plt.fill_between(np.arange(num_steps), np.mean(loss_curves, axis=0) - np.std(loss_curves, axis=0), np.mean(loss_curves, axis=0) + np.std(loss_curves, axis=0), alpha=0.5)
plt.show()
"""
# calculate the first index below threshold for each curve
fibt = np.apply_along_axis(first_index_below_threshold, 1, loss_curves, threshold)
"""
# plot the histogram of the first index below threshold
plt.hist(fibt, bins=100)
plt.show()
"""
lcs, fibts, mean_values, last_values = [], [], [], []
for switch_time in switch_times:
    lc = - 1 * eval_switcher_optimizer(problem_list, 
                                        optimizer_class_list, 
                                        num_steps, 
                                        config, 
                                        switch_time, 
                                        do_init_weights=False)

    lcs.append(lc)
    
    # calculate the first index below threshold for each curve
    fibt = np.apply_along_axis(first_index_below_threshold, 1, lc, threshold)
    fibts.append(fibt)

    #calulate the last value of the curve
    last_values.append(lc[:,-1])
    
    #calculate the mean value of the curve
    mean_values.append(np.mean(lc, axis=1))

    print(f'finished switch time {switch_time}')

# plot mean and std of loss curves for each switch time, 
# on the same plot, with legend
for i, switch_time in enumerate(switch_times):
    plt.plot(np.mean(lcs[i], axis=0), label=f'switch time {switch_time}')
    #plt.fill_between(np.arange(num_steps), np.mean(lcs[i], axis=0) - np.std(lcs[i], axis=0), np.mean(lcs[i], axis=0) + np.std(lcs[i], axis=0), alpha=0.5)

plt.legend(switch_times)
plt.show()


# plot fibt distribution for each switch time along x axis
for i, _ in enumerate(switch_times):
    plt.scatter(np.ones_like(fibts[i]) * switch_times[i], fibts[i], alpha=0.5)
    #add x axis label
    plt.xlabel('switch time')
    #add y axis label
    plt.ylabel('first index below threshold')
plt.legend()
# add title
plt.title('first index below threshold distribution for different switch times')
plt.show()

# plot the mean value of the curve for each switch time along x axis
for i, _ in enumerate(switch_times):
    plt.scatter(np.ones_like(mean_values[i]) * switch_times[i], mean_values[i], alpha=0.5)
    #add x axis label
    plt.xlabel('switch time')
    #add y axis label
    plt.ylabel('mean value')
plt.legend()
# add title
plt.title('mean value of the curve for different switch times')
plt.show()

# plot the last value of the curve for each switch time along x axis
for i, _ in enumerate(switch_times):
    plt.scatter(np.ones_like(last_values[i]) * switch_times[i], last_values[i], alpha=0.5)
    #add x axis label
    plt.xlabel('switch time')
    #add y axis label
    plt.ylabel('last value')
plt.legend()
# add title
plt.title('last value of the curve for different switch times')
plt.show()



