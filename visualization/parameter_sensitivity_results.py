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
#with open('visualization/all_score_matrices.pkl', 'rb') as f:
#   all_score_matrices = pickle.load(f)

# plot the results
plt.plot(agent_performance)
# plot the mean
plt.plot(np.mean(agent_performance, axis=0))
#add labels
plt.xlabel('number of timesteps')
plt.ylabel('number of times agent was best optimizer')
plt.show()
