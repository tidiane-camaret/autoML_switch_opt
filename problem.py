import torch 
from torch import nn
import numpy as np 
from torch.nn import functional as F
import torchvision.datasets as datasets

def init_weights(m): 
    # initialize weights of the model m 
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def mlp_problem():
    #print("here")
    num_vars = 2

    # Create four gaussian distributions with random mean and covariance
    gaussians = [
        torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.randn(num_vars),
            covariance_matrix=torch.eye(num_vars) * torch.rand(1),
            #scale_tril=torch.tril(torch.randn((num_vars, num_vars))),
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

    model0 = nn.Sequential(
        nn.Linear(num_vars, 2), nn.ReLU(), nn.Linear(2, 1), nn.Sigmoid()
    )

    model0.apply(init_weights)

    def obj_function(model):
        y_hat = model(x).view(-1)
        weight_norm = model[0].weight.norm() + model[2].weight.norm()
        return F.binary_cross_entropy(y_hat, y) + 5e-4 / 2 * weight_norm

    return {"model0": model0, "obj_function": obj_function, "dataset": (x, y)}

class MNISTProblemClass:
    def __init__(self, 
                 quantization_level=2,
                 ):
        
        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
        
        #maximum is 45 
        
        self.num_classes = 10
        #choose from the number of classes two random ones
        self.pair = np.random.choice(self.num_classes, 2)
        
        # Selecting the classes chosen 
        idx = (mnist_trainset.targets==self.pair[0]) | (mnist_trainset.targets==self.pair[1]) 
        mnist_trainset.targets = mnist_trainset.targets[idx]
        mnist_trainset.data = mnist_trainset.data[idx]
    
        self.dataset = mnist_trainset
        
        
        self.model0 = nn.Sequential(
            nn.Linear(num_vars, 2), nn.ReLU(), nn.Linear(2, self.num_classes), nn.Sigmoid()
        )

        self.model0.apply(init_weights)

        self.obj_function = self._obj_function
        
# define mlp_problem, but as a class
class MLPProblemClass:

    def __init__(self):
        num_vars = 2

        # Create four gaussian distributions with random mean and covariance
        gaussians = [
            torch.distributions.multivariate_normal.MultivariateNormal(
                loc=torch.randn(num_vars),
                covariance_matrix=torch.eye(num_vars) * torch.rand(1),
                #scale_tril=torch.tril(torch.randn((num_vars, num_vars))),
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