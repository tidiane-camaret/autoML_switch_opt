
import numpy as np
from problem import NormProblem, RosenbrockProblem, RastriginProblem
import pandas as pd
# open a csv file as pandas dataframe
df = pd.read_csv('graphs/all_math_inverse.csv')

#print data columns
print(df.columns)
df = df[["problem_train","problem_test", "agent_score"]]
print(df)
#plot a heatmap of scores
problems = df["problem_train"].unique()
heatmap = np.zeros((len(problems), len(problems)))
for i, problem_train in enumerate(problems):
    for j, problem_eval in enumerate(problems):
        heatmap[i,j] = df[(df["problem_train"] == problem_train) & (df["problem_test"] == problem_eval)]["agent_score"].mean()

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(heatmap, annot=True, fmt=".2f")

plt.show()

