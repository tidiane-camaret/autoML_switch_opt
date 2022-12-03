
import numpy as np
from problem import NormProblem, RosenbrockProblem, RastriginProblem
import pandas as pd
import matplotlib.pyplot as plt
# open a csv file as pandas dataframe
df = pd.read_csv('graphs/GtoN_all_opt.csv')

#print data columns
print(df.columns)
df = df[["agent_score","reward_system", "nb_timesteps"]]
print(df)

# for each reward system, plot mean and std of agent score over brackets of size 10000
reward_systems = df["reward_system"].unique()
fig, axs = plt.subplots(1, len(reward_systems), figsize=(20, 5))
for i, reward_system in enumerate(reward_systems):
    df_reward_system = df[df["reward_system"] == reward_system]
    #plot the individual points
    axs[i].scatter(df_reward_system["nb_timesteps"], df_reward_system["agent_score"], s=1)
    #plot the mean and std
    bracket_size = 10000
    nb_brackets = int(df["nb_timesteps"].max() / bracket_size)
    mean_scores = np.zeros(nb_brackets)
    std_scores = np.zeros(nb_brackets)
    for j in range(nb_brackets):
        mean_scores[j] = df_reward_system[(df_reward_system["nb_timesteps"] > j*bracket_size) & (df_reward_system["nb_timesteps"] < (j+1)*bracket_size)]["agent_score"].mean()
        std_scores[j] = df_reward_system[(df_reward_system["nb_timesteps"] > j*bracket_size) & (df_reward_system["nb_timesteps"] < (j+1)*bracket_size)]["agent_score"].std()
    axs[i].plot([bracket_size*(j+1) for j in range(nb_brackets)], mean_scores)
    axs[i].fill_between([bracket_size*(j+1) for j in range(nb_brackets)], mean_scores-std_scores, mean_scores+std_scores, alpha=0.5)
    # also plot the individual scores

    axs[i].set_title(reward_system)
plt.show()






    

    


"""
#plot a heatmap of scores
problems = df["reward_system"].unique()
heatmap = np.zeros((len(problems), len(problems)))
for i, problem_train in enumerate(problems):
    for j, problem_eval in enumerate(problems):
        heatmap[i,j] = df[(df["problemclass_train"] == problem_train) & (df["problemclass_eval"] == problem_eval)]["agent_score"].mean()

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(heatmap, annot=True, fmt=".2f")

plt.show()
"""
