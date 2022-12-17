import numpy as np
from problem import NormProblem, RosenbrockProblem, RastriginProblem
import pandas as pd
import matplotlib.pyplot as plt

nb_steps = [0, 25000, 50000, 75000, 100000]
perf = {
    "SGD": {"mean":[0.23, 0.25, 0.23, 0.20, 0.16],
            "std":[0.05, 0.05, 0.07, 0.05, 0.04]},
    "Adam": {"mean":[0.51, 0.52, 0.50, 0.40, 0.28],
            "std":[0.18, 0.13, 0.1, 0.07, 0.05]},
    "Agent": {"mean":[0.16, 0.23, 0.27, 0.40, 0.56],
            "std":[0.15, 0.12, 0.1, 0.07, 0.05]},
}

#plot mean and std of agent score 
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
plt.legend(perf.keys())
for i, agent in enumerate(perf.keys()):
    #plot the mean and std
    axs.plot(nb_steps, perf[agent]["mean"])
    axs.fill_between(nb_steps, np.array(perf[agent]["mean"])-np.array(perf[agent]["std"]), np.array(perf[agent]["mean"])+np.array(perf[agent]["std"]), alpha=0.3)
    # also plot the individual scores
#add the legend (for mean values only)
#axs.legend(perf.keys())

axs.set_title("Actions taken by the agent on MNIST", fontsize=16)
#add title for axis
axs.set_xlabel("Training steps", fontsize=16)
axs.set_ylabel("Starting point", fontsize=16)

plt.show()

"""

labels = ["lookahead", "inverse", "threshold", "opposite"]

res = {"same":
        {"mean":[0.33, 0.17, 0.16, 0.23],
        "std":[0.1, 0.09, 0.12, 0.13,]}
        ,
        "gen":
        {"mean":[0.07, 0.27, 0.14, 0.19],
        "std":[0.04, 0.07, 0.05, 0.07,]}
}

#plot a groupe bar chart of gen and same, with mean and std
fig, axs = plt.subplots(1, 1, figsize=(10, 5))
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

rects1 = axs.bar(x - width/2, res["same"]["mean"], width, yerr=res["same"]["std"], label='Same for train and eval')
rects2 = axs.bar(x + width/2, res["gen"]["mean"], width, yerr=res["gen"]["std"], label='Generalization')

# Add some text for labels, title and custom x-axis tick labels, etc.
axs.set_ylabel('Scores', fontsize=16)
axs.set_title
axs.set_xticks(x)
axs.set_xticklabels(labels, fontsize=16)
axs.legend()

fig.tight_layout()

plt.show()

"""