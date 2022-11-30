# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:42:40 2022

@author: DELL
"""

import numpy as np
from problem import NormProblem, RosenbrockProblem, RastriginProblem
import pandas as pd
# open a csv file as pandas dataframe
df = pd.read_csv('graphs/wandb_export_2022-11-30T16 44 19.974+01 00.csv')

#print data columns
print(df.columns)
df = df[["problemclass_train","problemclass_eval", "agent_score"]]
print(df)
#plot a heatmap of scores
problems = df["problemclass_train"].unique()
heatmap = np.zeros((len(problems), len(problems)))
for i, problem_train in enumerate(problems):
    for j, problem_eval in enumerate(problems):
        heatmap[i,j] = df[(df["problemclass_train"] == problem_train) & (df["problemclass_eval"] == problem_eval)]["agent_score"].mean()

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(heatmap, annot=True, fmt=".2f")

plt.show()

