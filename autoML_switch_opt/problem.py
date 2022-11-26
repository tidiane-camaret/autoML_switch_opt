import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, RandomSampler, Subset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

    
    
def init_weights(m):
    # initialize weights of the model m
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class MNISTProblemClass:
    
    def __init__(self, 
                 classes,
                 batch_size=5,
                 num_classes_selected=2,
                 weights_flag = False
                 ):
        
        
        
        
        #mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
        
        #maximum is 45 
        self.batch_size = batch_size
        self.num_classes_selected= num_classes_selected
        self.weights_flag = weights_flag
        self.num_classes = 10
        #classes that define the problem
        self.classes = classes
        #tranform data, resize to make it smaller
        transform =  transforms.Compose([transforms.ToTensor(), transforms.Resize(10),
                                            transforms.Normalize((0.5,), (0.5,))
                                            ])
        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        
        # Selecting the classes chosen  from train dataset
        idx = (mnist_trainset.targets==self.classes[0]) | (mnist_trainset.targets==self.classes[1]) 
        
        mnist_trainset.targets = mnist_trainset.targets[idx]
        mnist_trainset.data = mnist_trainset.data[idx]
        
        #assign classes to 0 or 1 label - for proper computing of loss
        for i in range(0, len(mnist_trainset.targets )):
            if mnist_trainset.targets[i] ==self.classes[0]:
                
                mnist_trainset.targets[i] = 0
            else:
                
                mnist_trainset.targets[i] = 1
        
        #set dataset
        self.dataset = mnist_trainset
        
        
        #model definition
        
        
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
        
        
        
        
        if(weights_flag):
            self.model0.apply(init_weights)
        
        

        self.obj_function = self._obj_function
        
    
    #prepares data from sampler to be inputted to model 
    def load_data_from_sampler(self, subset):
        batch_data = torch.tensor([])
        batch_labels = torch.tensor([])
        #loop over the batch generated,
        #data samples go in batch_data tensor, labels go in batch_labels
        for data, target in subset:
            batch_data = torch.cat((batch_data, data), dim = 0)
            batch_labels = torch.cat((batch_labels, torch.tensor([int(target)])), dim = 0)
        return batch_data, batch_labels
    
    
    
    def _obj_function(self, model):
        #defining criteria of loss
        criterion = nn.BCELoss()
        #use random sampler to get random indices in dataset
        #the indices will determine our random batch
        mnist_trainset = self.dataset
        #get defining indices
        batch_indices = RandomSampler(mnist_trainset, replacement=True, num_samples=self.batch_size, generator=None)
        #pass indices to subset in order to get a sample
        current_batch = Subset(mnist_trainset, list(batch_indices))
        running_loss=0
        #split the subsample into data and labels to be fed to model
        batch_data, batch_labels = self.load_data_from_sampler(current_batch)
        #reshaping bc pytorch wants input of [batch, channel, height, width] - since we have o
        batch_data = batch_data[:, None, :, :]
        
        y_hat = model(batch_data)
        batch_labels = torch.reshape(batch_labels, ( 5, 1))
        
        loss = criterion(y_hat, batch_labels)
        running_loss += loss.item()
        self.running_loss = running_loss
        return loss
        
    def get_obj_function(self):
        return self.obj_function


class Variable2D(nn.Module):
    """A wrapper to turn a tensor of parameters into a module for optimization."""

    def __init__(self, data: torch.Tensor):
        """Create Variable holding `data` tensor."""
        super().__init__()
    
        self.x = nn.Parameter(data[0])
        
        self.y = nn.Parameter(data[1])
        
class Variable1D(nn.Module):
    """A wrapper to turn a tensor of parameters into a module for optimization."""

    def __init__(self, data: torch.Tensor):
        """Create Variable holding `data` tensor."""
        super().__init__()
        self.x = nn.Parameter(data)

# define mlp_problem, but as a class



class MLPProblemClass:

    def __init__(self):
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

class Beale:
    def __init__(self, x_start, y_start):
        
        self.model0 = Variable2D(torch.tensor([x_start, y_start],  dtype=torch.float32, requires_grad= True))
    
    # def derivative(self):
    #     #calculate partial gradients
    #     x = self.x
    #     y = self.y
    #     partial_x = 2*(1.5-x+ (x*y))*-1*y + 2*(2.25-x+(x*y)**2)* (y**2) + 2*(2.625 - x + (x*y)**3)*(y**3)
    #     partial_y = 2*(1.5-x+ (x*y))*y + 2*(2.25-x+(x*y)**2)*x*2*y + 2*(2.625 - x + (x*y)**3)*(3*x*(y**2))
        
    #     return partial_x, partial_y
    
    def function(self, model):
        x = model.x
        y = model.y
        #print(model.x)
        first_comp = (1.5 - x+ (x*y))**2
        second_comp = (2.25 - x+ (x*(y**2)))**2
        third_comp = (2.625 - x + (x*(y)**3))**2
        
        result = first_comp + second_comp + third_comp
        return result
    
    def obj_function(self, model):
        return self.function(model)
        
    
    
    
    


                       
        
        