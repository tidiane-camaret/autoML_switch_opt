from problem import mlp_problem
from model import buildmodel
import torch
from environment import *
import stable_baselines3
from train_agent import eval_agent
import numpy as np
import matplotlib.pyplot as plt

# define our inputs and outputs
x, y = mlp_problem()

# define our model
model = buildmodel()
