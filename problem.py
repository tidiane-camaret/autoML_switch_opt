import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, RandomSampler, Subset
import torchvision.transforms as transforms

device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

def init_weights(m):
    # initialize weights of the model m
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
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
                num_samples=25,):

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
        self.model0.device = device
        print('print device is :  ',self.model.device)

        self.model0.apply(init_weights)

        self.dataset = (x, y)

    def obj_function(self, model):
        x, y = self.dataset
        y_hat = model(x).view(-1)
        weight_norm = model[0].weight.norm() + model[2].weight.norm()
        return F.binary_cross_entropy(y_hat, y) + 5e-4 / 2 * weight_norm



# define rosenbrock problem as a class
class RosenbrockProblem:
    
        def __init__(self, 
                    x0=None,
                    num_vars=2,     
                    ):
            if x0 is None:
                x0 = torch.tensor([-1.5 if i % 2 == 0 else 1.5 for i in range(num_vars)], dtype=torch.float32, requires_grad=True)
            else :
                x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
                
            self.model0 = Variable(x0)
            self.tuned_lrs = {torch.optim.Adam: 0.01,
                    torch.optim.SGD: 0.01,
                    torch.optim.RMSprop: 0.01,
        }

        #def function_def(self, x):
        #    return torch.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

        def function_def(self, x, y):
            return (1-x)**2 + 100*(y-x**2)**2

        def obj_function(self, model):
            x = model.x
            return self.function_def(x[0],x[1])
    
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
            self.tuned_lrs = {torch.optim.Adam: 0.01,
                    torch.optim.SGD: 0.01,
                    torch.optim.RMSprop: 0.01,
        }


        def function_def(self, x):
            return self.scale*(x-self.center)**2

        def obj_function(self, model):
            x = model.x
            return self.function_def(x[0])

class NoisyHillsProblem:

        def __init__(self, 
                x0=[0,0],
                scale=1,
                center=1
                ):

            x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
            self.model0 = Variable(x0)
            self.scale = scale
            self.center = center
            self.tuned_lrs = {torch.optim.Adam: 0.01,
                    torch.optim.SGD: 0.01,
                    torch.optim.RMSprop: 0.01,
        }


        def function_def(self, x, y):
            return -1 * torch.sin(x * x) * torch.cos(3 * y * y) * torch.exp(-(x * y) * (x * y)) - torch.exp(-(x + y) * (x + y))


        def obj_function(self, model):
            x = model.x
            return self.function_def(x[0],x[1])
           


class RastriginProblem():

    def __init__(self, 
                x0=[0,0],
                scale=1,
                center=1
                ):

        x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        self.model0 = Variable(x0)
        self.scale = scale
        self.center = center
        self.tuned_lrs = {torch.optim.Adam: 0.01,
                    torch.optim.SGD: 0.01,
                    torch.optim.RMSprop: 0.01,
        }


    def function_def(self, x, y):
        return 20 + x**2 - 10*torch.cos(2*np.pi*x) + y**2 - 10*torch.cos(2*np.pi*y)

    def obj_function(self, model):
        x = model.x
        return self.function_def(x[0],x[1])


class AckleyProblem():

    def __init__(self, 
                x0=[0,0],
                scale=1,
                center=1
                ):

        x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        self.model0 = Variable(x0)
        self.scale = scale
        self.center = center
        self.tuned_lrs = {torch.optim.Adam: 0.01,
                    torch.optim.SGD: 0.01,
                    torch.optim.RMSprop: 0.01,
        }


    def function_def(self, x, y):
        return -20*torch.exp(-0.2*torch.sqrt(0.5*(x**2 + y**2))) - torch.exp(0.5*(torch.cos(2*torch.pi*x) + torch.cos(2*torch.pi*y))) + np.exp(1) + 20

    def obj_function(self, model):
        x = model.x
        return self.function_def(x[0],x[1])
class GaussianHillsProblem:

    def __init__(self, 
            x0=[0,0],
            scale=1,
            center=1
            ):

        x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        self.model0 = Variable(x0)
        self.scale = scale
        self.center = center            
        self.tuned_lrs = {torch.optim.Adam: 0.01,
                    torch.optim.SGD: 0.01,
                    torch.optim.RMSprop: 0.01,
        }


    def fd(self, x, y, x_mean, y_mean, x_sig, y_sig):
        normalizing = 1 / (2 * torch.pi * x_sig * y_sig)
        x_exp = (-1 * (x - x_mean)**2) / (2 * (x_sig)**2)
        y_exp = (-1 * (y - y_mean)**2) / (2 * (y_sig)**2)
        return normalizing * torch.exp(x_exp + y_exp)

    def function_def(self, x, y):

        z = -1 * self.fd(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.35, y_sig=0.35)


        # three steep gaussian trenches
        z -= self.fd(x, y, x_mean=1.0, y_mean=-0.5, x_sig=0.1, y_sig=0.5)
        z -= self.fd(x, y, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.2)
        z -= self.fd(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)

        return z


    def obj_function(self, model):
        x = model.x
        return self.function_def(x[0],x[1])

class NormProblem:

    def __init__(self, 
            x0=[0,0],
            scale=1,
            center=1
            ):

        x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        self.model0 = Variable(x0)
        self.scale = scale
        self.center = center
        self.tuned_lrs = {torch.optim.Adam: 0.01,
                    torch.optim.SGD: 0.01,
                    torch.optim.RMSprop: 0.01,
        }


    def function_def(self, x, y):
        return torch.sqrt(x**2 + (1.1*y)**2)
    def obj_function(self, model):
        x = model.x
        return self.function_def(x[0],x[1])

class YNormProblem:

    def __init__(self, 
            x0=[0,0],
            scale=1,
            center=1
            ):

        x0 = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        self.model0 = Variable(x0)
        self.scale = scale
        self.center = center
        self.tuned_lrs = {torch.optim.Adam: 0.01,
                    torch.optim.SGD: 0.01,
                    torch.optim.RMSprop: 0.01,
        }


    def function_def(self, x, y):
        return torch.sqrt(y**2) * 10

    def obj_function(self, model):
        x = model.x
        return self.function_def(x[0],x[1])

class MNISTProblemClass:

    def __init__(self,
                 classes,
                 batch_size=5,
                 num_classes_selected=2,
                 weights_flag=False
                 ):


        # maximum is 45
        self.batch_size = batch_size
        self.num_classes_selected = num_classes_selected
        self.weights_flag = weights_flag
        self.num_classes = 10
        # classes that define the problem
        self.classes = classes
        # tranform data, resize to make it smaller
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(10),
                                        transforms.Normalize((0.5,), (0.5,))
                                        ])
        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

        # Selecting the classes chosen  from train dataset
        idx = (mnist_trainset.targets == self.classes[0]) | (mnist_trainset.targets == self.classes[1])

        mnist_trainset.targets = mnist_trainset.targets[idx]
        mnist_trainset.data = mnist_trainset.data[idx]

        # assign classes to 0 or 1 label - for proper computing of loss
        for i in range(0, len(mnist_trainset.targets)):
            if mnist_trainset.targets[i] == self.classes[0]:

                mnist_trainset.targets[i] = 0
            else:

                mnist_trainset.targets[i] = 1

        # set dataset
        self.dataset = mnist_trainset

        # model definition

        self.model0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=5,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),

            nn.AdaptiveMaxPool2d(3),
            nn.Flatten(),

            nn.Linear(45, 1),
            nn.Sigmoid()
        )

        if (weights_flag):
            self.model0.apply(init_weights)

        self.obj_function = self._obj_function

        self.tuned_lrs = {torch.optim.Adam: 0.05,
                            torch.optim.SGD: 0.1,
                            torch.optim.RMSprop: 0.01,
        }

    # prepares data from sampler to be inputted to model
    def load_data_from_sampler(self, subset):
        batch_data = torch.tensor([])
        batch_labels = torch.tensor([])
        # loop over the batch generated,
        # data samples go in batch_data tensor, labels go in batch_labels
        for data, target in subset:
            batch_data = torch.cat((batch_data, data), dim=0)
            batch_labels = torch.cat((batch_labels, torch.tensor([int(target)])), dim=0)
        return batch_data, batch_labels

    def _obj_function(self, model):
        # defining criteria of loss
        criterion = nn.BCELoss()
        # use random sampler to get random indices in dataset
        # the indices will determine our random batch
        mnist_trainset = self.dataset
        # get defining indices
        batch_indices = RandomSampler(mnist_trainset, replacement=True, num_samples=self.batch_size, generator=None)
        # pass indices to subset in order to get a sample
        current_batch = Subset(mnist_trainset, list(batch_indices))
        running_loss = 0
        # split the subsample into data and labels to be fed to model
        batch_data, batch_labels = self.load_data_from_sampler(current_batch)
        # reshaping bc pytorch wants input of [batch, channel, height, width] - since we have o
        batch_data = batch_data[:, None, :, :]

        y_hat = model(batch_data)
        batch_labels = torch.reshape(batch_labels, (5, 1))

        loss = criterion(y_hat, batch_labels)
        running_loss += loss.item()
        self.running_loss = running_loss
        return loss

    def get_obj_function(self):
        return self.obj_function


class ImageDatasetProblemClass:

    def __init__(self,
                 classes,
                 dataset_class = datasets.FashionMNIST,
                 batch_size=5,
                 num_classes_selected=2,
                 weights_flag=False
                 ):


        # maximum is 45
        self.batch_size = batch_size
        self.num_classes_selected = num_classes_selected
        self.weights_flag = weights_flag
        self.num_classes = 10
        # classes that define the problem
        self.classes = classes
        # tranform data, resize to make it smaller
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(10),
                                        transforms.Normalize((0.5,), (0.5,))
                                        ])
        mnist_trainset = dataset_class(root='./data', train=True, download=True, transform=transform)

        # Selecting the classes chosen  from train dataset
        idx = (mnist_trainset.targets == self.classes[0]) | (mnist_trainset.targets == self.classes[1])

        mnist_trainset.targets = mnist_trainset.targets[idx]
        mnist_trainset.data = mnist_trainset.data[idx]

        # assign classes to 0 or 1 label - for proper computing of loss
        for i in range(0, len(mnist_trainset.targets)):
            if mnist_trainset.targets[i] == self.classes[0]:

                mnist_trainset.targets[i] = 0
            else:

                mnist_trainset.targets[i] = 1

        # set dataset
        self.dataset = mnist_trainset

        # model definition

        self.model0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=5,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),

            nn.AdaptiveMaxPool2d(3),
            nn.Flatten(),

            nn.Linear(45, 1),
            nn.Sigmoid()
        )

        if (weights_flag):
            self.model0.apply(init_weights)

        self.obj_function = self._obj_function

        self.tuned_lrs = {torch.optim.Adam: 0.05,
                            torch.optim.SGD: 0.1,
                            torch.optim.RMSprop: 0.01,
        }

    # prepares data from sampler to be inputted to model
    def load_data_from_sampler(self, subset):
        batch_data = torch.tensor([])
        batch_labels = torch.tensor([])
        # loop over the batch generated,
        # data samples go in batch_data tensor, labels go in batch_labels
        for data, target in subset:
            batch_data = torch.cat((batch_data, data), dim=0)
            batch_labels = torch.cat((batch_labels, torch.tensor([int(target)])), dim=0)
        return batch_data, batch_labels

    def _obj_function(self, model):
        # defining criteria of loss
        criterion = nn.BCELoss()
        # use random sampler to get random indices in dataset
        # the indices will determine our random batch
        mnist_trainset = self.dataset
        # get defining indices
        batch_indices = RandomSampler(mnist_trainset, replacement=True, num_samples=self.batch_size, generator=None)
        # pass indices to subset in order to get a sample
        current_batch = Subset(mnist_trainset, list(batch_indices))
        running_loss = 0
        # split the subsample into data and labels to be fed to model
        batch_data, batch_labels = self.load_data_from_sampler(current_batch)
        # reshaping bc pytorch wants input of [batch, channel, height, width] - since we have o
        batch_data = batch_data[:, None, :, :]

        y_hat = model(batch_data)
        batch_labels = torch.reshape(batch_labels, (5, 1))

        loss = criterion(y_hat, batch_labels)
        running_loss += loss.item()
        self.running_loss = running_loss
        return loss

    def get_obj_function(self):
        return self.obj_function
