import torch
from torch import nn
import numpy as np


def mlp_problem():
    num_vars = 2

    # Create four gaussian distributions with random mean and covariance
    gaussians = [
        torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.randn(num_vars),
            covariance_matrix=torch.eye(num_vars) * torch.rand(1),
            # scale_tril=torch.tril(torch.randn((num_vars, num_vars))),
        )
        for _ in range(4)
    ]

    # Randomly assign each of the four gaussians a 0-1 label
    # Do again if all four gaussians have the same label (don't want that)
    gaussian_labels = np.zeros((4,))
    while (gaussian_labels == 0).all() or (gaussian_labels == 1).all():
        gaussian_labels = torch.randint(0, 2, size=(4,))

    # Generate a dataset of 100 points with 25 points drawn from each gaussian
    # Label of the datapoint is the same as the label of the gaussian it came from
    x = torch.cat([g.sample((25,)) for g in gaussians])
    y = torch.cat([torch.full((25,), float(label)) for label in gaussian_labels])
    perm = torch.randperm(len(x))
    x = x[perm]
    y = y[perm]

    return x, y
