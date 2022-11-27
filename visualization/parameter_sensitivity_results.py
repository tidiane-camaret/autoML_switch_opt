import pickle
import numpy as np
import matplotlib.pyplot as plt

# load the results
with open('visualization/agent_performance.pkl', 'rb') as f:
    agent_performance = pickle.load(f)
# load all_optimizer_results
with open('visualization/all_optimizer_results.pkl', 'rb') as f:
    all_optimizer_results = pickle.load(f)
# load all_score_matrices
with open('visualization/all_score_matrices.pkl', 'rb') as f:
   all_score_matrices = pickle.load(f)

#agent_performance = agent_performance[:, 8]
all_score_matrices = np.asarray(all_score_matrices)

optimizer_class_list = list(all_optimizer_results[0][0].keys())

# function that takes in a list and returns a list of the same length 
# with 1 at the index of the maximum value and 0 elsewhere
def top(l):
    min_index = np.argmin(l)
    new_l = np.zeros(len(l))
    new_l[min_index] = 1
    return new_l

# function that takes in a list and divides each element by its max
def normalize(l):
    min_val = np.min(l)
    return [x / min_val for x in l]

print(agent_performance.shape)
top_matrices = np.apply_along_axis(top, 3, all_score_matrices)
top_matrices = np.mean(top_matrices, axis=2)
top_matrices_mean = np.mean(top_matrices, axis=1)
top_matrices_std = np.std(top_matrices, axis=1)
print(top_matrices_mean)
print(top_matrices_std)

normalized_matrices = np.apply_along_axis(normalize, 3, all_score_matrices)
normalized_matrices = np.mean(normalized_matrices, axis=2)
normalized_matrices_mean = np.mean(normalized_matrices, axis=1)
normalized_matrices_std = np.std(normalized_matrices, axis=1)
print(normalized_matrices_mean)
print(normalized_matrices_std)


fig, ax = plt.subplots(1,1,figsize=(10, 10))
# plot mean and std of top_matrices
for i in range(top_matrices_mean.shape[1]):
    ax.plot( top_matrices_mean[:, i], label=optimizer_class_list[i])
    # add error bars
    ax.fill_between(np.arange(top_matrices_mean.shape[0]),
                        top_matrices_mean[:, i] - top_matrices_std[:, i],
                        top_matrices_mean[:, i] + top_matrices_std[:, i],
                        alpha=0.2)
    ax.set_xlabel("nb of agent training steps")
ax.legend()


"""
# plot mean and std of normalized_matrices
for i in range(normalized_matrices_mean.shape[1]):
    ax[1].plot(normalized_matrices_mean[:, i], label="optimizer {}".format(i))
    # add error bars
    ax[1].fill_between(np.arange(normalized_matrices_mean.shape[0]),
                        normalized_matrices_mean[:, i] - normalized_matrices_std[:, i],
                        normalized_matrices_mean[:, i] + normalized_matrices_std[:, i],
                        alpha=0.2)
ax[1].legend()
"""
#save plot
plt.savefig('visualization/parameter_sensitivity_results.png')
