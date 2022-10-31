import copy
from scipy.optimize import minimize, rosen, rosen_der
import numpy as np
from torch import nn
import torch
import scipy

def run_optimizer(make_optimizer, problem, iterations, hyperparams):
    # Initial solution
    model = copy.deepcopy(problem["model0"])
    obj_function = problem["obj_function"]

    # Define optimizer
    optimizer = make_optimizer(model.parameters(), **hyperparams)

    # We will keep track of the objective values and weight trajectories
    # throughout the optimization process.
    values = []
    trajectory = []

    # Passed to optimizer. This setup is required to give the autonomous
    # optimizer access to the objective value and not just its gradients.
    def closure():
        trajectory.append(copy.deepcopy(model))
        optimizer.zero_grad()

        obj_value = obj_function(model)
        obj_value.backward()

        values.append(obj_value.item())
        return obj_value

    # Minimize
    for i in range(iterations):
        optimizer.step(closure)

        # Stop optimizing if we start getting nans as objective values
        if np.isnan(values[-1]) or np.isinf(values[-1]):
            break

    return np.nan_to_num(values, 1e6), trajectory

class Variable(nn.Module):
    """A wrapper to turn a tensor of parameters into a module for optimization."""

    def __init__(self, data: torch.Tensor):
        """Create Variable holding `data` tensor."""
        super().__init__()
        self.x = nn.Parameter(data)


def convex_quadratic():
    """
    Generate a symmetric positive semidefinite matrix A with eigenvalues
    uniformly in [1, 30].

    """
    num_vars = 2

    # First generate an orthogonal matrix (of eigenvectors)
    eig_vecs = torch.tensor(
        scipy.stats.ortho_group.rvs(dim=(num_vars)), dtype=torch.float
    )
    # Now generate eigenvalues
    eig_vals = torch.rand(num_vars) * 29 + 1

    A = eig_vecs @ torch.diag(eig_vals) @ eig_vecs.T
    b = torch.normal(0, 1 / np.sqrt(num_vars), size=(num_vars,))

    x0 = torch.normal(0, 0.5 / np.sqrt(num_vars), size=(num_vars,))

    def quadratic(var):
        x = var.x
        return 0.5 * x.T @ A @ x + b.T @ x

    optimal_x = scipy.linalg.solve(A.numpy(), -b.numpy(), assume_a="pos")
    optimal_val = quadratic(Variable(torch.tensor(optimal_x))).item()

    return {
        "model0": Variable(x0),
        "obj_function": quadratic,
        "optimal_x": optimal_x,
        "optimal_val": optimal_val,
        "A": A.numpy(),
        "b": b.numpy(),
    }

def rosenbrock():
    num_vars = 2

    # Initialization strategy: x_i = -2 if i is even, x_i = +2 if i is odd
    x0 = torch.tensor([-1.5 if i % 2 == 0 else 1.5 for i in range(num_vars)])

    def rosen(var):
        x = var.x
        return torch.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

    # Optimum at all x_i = 1, giving f(x) = 0
    optimal_x = np.ones(num_vars)
    optimal_val = 0

    return {
        "model0": Variable(x0),
        "obj_function": rosen,
        "optimal_x": optimal_x,
        "optimal_val": optimal_val,
    }



def minimize_custom_():
    #https://github.com/sdamadi/mathelecs/blob/main/01_Minimizing_every_function_using_Pytorch.ipynb
    x_0 = torch.tensor(0., requires_grad = True)
    x = x_0
    optimizer = torch.optim.SGD([x], lr=0.1)
    steps = 30
    for i in range(steps):
        optimizer.zero_grad()
        f = (x-1)**2
        f.backward()
        optimizer.step()
        print(f'At step {i+1:2} the function value is {f.item(): 1.4f} and x={x: 0.4f}' )


def minimize_custom(objective,
                    optimizer_class=torch.optim.SGD,
                    x_0=0.,
                    lr=0.1,
                    steps=30):

    x_t = copy.deepcopy(torch.tensor(x_0, requires_grad = True))
    optimizer = optimizer_class([x_t], lr=lr)
    X, F = [], []
    

    for i in range(steps):
        optimizer.zero_grad()
        f = objective(x_t)
        f.backward()
        optimizer.step()
        X.append(x_t.item())
        F.append(objective(x_t.item()))

        #print(f'At step {i+1:2} the function value is {f.item(): 1.4f} and x={x_t: 0.4f}' )
    return X,F


def norm_function(x):
        return (x-1)**2

def rosen_function(x):
    return torch.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
