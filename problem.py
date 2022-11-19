import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


def init_weights(m):
    # initialize weights of the model m
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Variable(nn.Module):
    """A wrapper to turn a tensor of parameters into a module for optimization."""

    def __init__(self, data: torch.Tensor):
        """Create Variable holding `data` tensor."""
        super().__init__()
        self.x = nn.Parameter(data)


# define mlp_problem, but as a class
class MLPProblemClass:

    def __init__(self,
                 num_vars=2,
                 num_gaussians=4,
                 num_samples=25, ):

        # generate list of random covariance matrices

        trils = []
        for _ in range(num_gaussians):
            mat = torch.rand(num_vars, num_vars)
            mat = mat + 2 * torch.diag_embed(torch.absolute(torch.diag(mat)))
            tril = torch.tril(mat)
            trils.append(tril)

        # Create gaussian distributions with random mean and covariance
        gaussians = [
            torch.distributions.multivariate_normal.MultivariateNormal(
                loc=torch.randn(num_vars),
                # covariance_matrix=cov[i]
                # covariance_matrix=torch.eye(num_vars) * torch.rand(1),
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

        self.dataset = (x, y)

    def obj_function(self, model):
        x, y = self.dataset
        y_hat = model(x).view(-1)
        weight_norm = model[0].weight.norm() + model[2].weight.norm()
        return F.binary_cross_entropy(y_hat, y) + 5e-4 / 2 * weight_norm


# define rosenbrock problem as a class
class RosenbrockProblemClass:
    def __init__(self,
                 x0=None,
                 num_vars=2,
                 ):
        if x0 is None:
            x0 = torch.tensor([-1.5 if i % 2 == 0 else 1.5 for i in range(num_vars)])
        else:
            x0 = torch.tensor(x0)

        self.model0 = Variable(x0)

    def function_def(self, x):
        return torch.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    def obj_function(self, model):
        x = model.x
        return self.function_def(x)


class SquareProblemClass:

    def __init__(self,
                 x0=0,
                 scale=1,
                 center=1
                 ):
        x0 = torch.tensor([x0], dtype=torch.float32, requires_grad=True)
        self.model0 = Variable(x0)
        self.scale = scale
        self.center = center

    def function_def(self, x):
        return self.scale * (x - self.center) ** 2

    def obj_function(self, model):
        x = model.x
        return self.function_def(x[0])

