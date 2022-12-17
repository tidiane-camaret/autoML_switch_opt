
import sys
sys.path.append('..')
from problem import NoisyHillsProblem, GaussianHillsProblem, RosenbrockProblem\
    ,RastriginProblem, SquareProblemClass, AckleyProblem
from agent_gen import agent_gen
import numpy as np



problem_list = [NoisyHillsProblem, GaussianHillsProblem, RosenbrockProblem, RastriginProblem, AckleyProblem]
nb_training_seqs = 10


#matrix of agent scores for each problem pair
agent_scores = np.zeros((len(problem_list), len(problem_list), nb_training_seqs))
"""
for i in range(len(problem_list)):
    for j in range(len(problem_list)):
        agent_scores[i,j,:] = np.asarray(agent_gen(problem_list[i], problem_list[j], nb_training_seqs))
        print("Problem pair ", problem_list[i].__name__, " ",problem_list[j].__name__, " done")
        print(np.mean(agent_scores[i,j,:]))

        #save the matrix
        np.save("agent_scores.npy", agent_scores)

"""

#load the matrix
agent_scores = np.load("agent_scores.npy")



#plot average agent scores, with titled legend
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
fig, ax = plt.subplots()
ax = sns.heatmap(agent_scores.mean(axis=2), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
ax.set_title("Average Agent Scores")
ax.set_xlabel("Problem 2")
ax.set_ylabel("Problem 1")
ax.set_xticklabels([problem.__name__ for problem in problem_list], rotation=45)
ax.set_yticklabels([problem.__name__ for problem in problem_list], rotation=0)
plt.show()



fig, ax = plt.subplots()
ax = sns.heatmap(agent_scores.max(axis=2), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
ax.set_title("Max Agent Scores")
ax.set_xlabel("Problem 2")
ax.set_ylabel("Problem 1")
ax.set_xticklabels([problem.__name__ for problem in problem_list], rotation=45)
ax.set_yticklabels([problem.__name__ for problem in problem_list], rotation=0)
plt.show()


#plot all scores trajectories on subplots
fig, axs = plt.subplots(len(problem_list), len(problem_list), sharex=True, sharey=True)
for i in range(len(problem_list)):
    for j in range(len(problem_list)):
        axs[i,j].plot(agent_scores[i,j,:])
        axs[i,j].set_title(str(i)+ "_" + str(j))
        axs[i,j].set_xlabel("Training sequence")
        axs[i,j].set_ylabel("Agent score")
plt.show()






