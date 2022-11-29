# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 15:42:40 2022

@author: DELL
"""

import torch
from problem import NormProblem, RosenbrockProblem, RastriginProblem

problem_list = [NormProblem, RosenbrockProblem, RastriginProblem]
print("Problem list: ", problem_list)
problem_list.remove(NormProblem)
print("Problem list: ", problem_list)
