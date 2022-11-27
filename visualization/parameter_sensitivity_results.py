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

agent_performance = agent_performance[:, 8]

# plot mean and std results
plt.plot(agent_performance.mean(axis=0))
plt.fill_between(np.arange(agent_performance.shape[1]),
                    agent_performance.mean(axis=0) - agent_performance.std(axis=0), 
                    agent_performance.mean(axis=0) + agent_performance.std(axis=0),
                    alpha=0.2)
plt.show()

#add labels
plt.xlabel('number of timesteps')
plt.ylabel('number of times agent was best optimizer')
plt.show()
plt.savefig('visualization/agent_performance.png')