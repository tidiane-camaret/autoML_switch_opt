import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


def init_weights(m):
    # initialize weights of the model m
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# define mlp_problem, but as a class
class MLPProblemClass:

    def __init__(self, 
                num_vars=2,
                num_gaussians=4,
                num_samples=25,):

        # generate list of random covariance matrices
        covs = []
        for _ in range(num_gaussians):
            cov = torch.rand(num_vars, num_vars)
            cov = torch.mm(cov, cov.t())
            cov.add_(torch.eye(num_vars))
            covs.append(cov)
        trils = []
        for _ in range(num_gaussians):
            mat = torch.rand(num_vars, num_vars)
            mat = mat + 2 * torch.diag_embed(torch.absolute(torch.diag(mat)))
            tril = torch.tril(mat)
            trils.append(tril)

        # Create four gaussian distributions with random mean and covariance
        gaussians = [
            torch.distributions.multivariate_normal.MultivariateNormal(
                loc=torch.randn(num_vars),
                #covariance_matrix=cov[i]
                #covariance_matrix=torch.eye(num_vars) * torch.rand(1),
                scale_tril=trils[i],
            )
            for i in range(num_gaussians)
        ]

        # Randomly assign each of the four gaussians a 0-1 label
        # Do again if all four gaussians have the same label (don't want that)
        gaussian_labels = np.zeros((num_gaussians,))
        while (gaussian_labels == 0).all() or (gaussian_labels == 1).all():
            gaussian_labels = torch.randint(0, 2, size=(num_gaussians,))

        # Generate a dataset of 100 points with 25 points drawn from each gaussian
        # Label of the datapoint is the same as the label of the gaussian it came from
        x = torch.cat([g.sample((num_samples,)) for g in gaussians])
        y = torch.cat([torch.full((num_samples,), float(label)) for label in gaussian_labels])
        perm = torch.randperm(len(x))
        x = x[perm]
        y = y[perm]

        self.model0 = nn.Sequential(
            nn.Linear(num_vars, 2), nn.ReLU(), nn.Linear(2, 1), nn.Sigmoid()
        )

        self.model0.apply(init_weights)

        self.obj_function = self._obj_function
        self.dataset = (x, y)

    def _obj_function(self, model):
        x, y = self.dataset
        y_hat = model(x).view(-1)
        weight_norm = model[0].weight.norm() + model[2].weight.norm()
        return F.binary_cross_entropy(y_hat, y) + 5e-4 / 2 * weight_norm

    def get_model0(self):
        return self.model0

    def get_obj_function(self):
        return self.obj_function

    def get_dataset(self):
        return self.dataset
