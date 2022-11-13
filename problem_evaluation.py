from problem import MLPProblemClass
from environment import eval_handcrafted_optimizer
import torch
from omegaconf import OmegaConf

config = OmegaConf.load('config.yaml')
num_problems = 1000

problem_list = [MLPProblemClass() for _ in range(num_problems)]

eval_handcrafted_optimizer(problem_list, torch.optim.SGD, 1000, config, do_init_weights=False)