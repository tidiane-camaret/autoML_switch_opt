import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn import Parameter
from torchvision import datasets, transforms

class MNIST_MLP2(nn.Module):

    def __init__(self, input_size =784, output_size=10, layers=[120, 84]):
        super(MNIST_MLP2, self).__init__()
        self.d1 = nn.Linear(input_size, layers[0])
        self.d2 = nn.Linear(layers[0], layers[1])
        self.d3 = nn.Linear(layers[1], output_size)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = self.d3(x)
        return F.log_softmax(x, dim=1)

class MNIST_CNN(nn.Module):
    """Custom module for a simple convnet classifier"""

    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # input is 28x28x1
        # conv1(kernel=5, filters=10) 28x28x10 -> 24x24x10
        # max_pool(kernel=2) 24x24x10 -> 12x12x10

        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # conv2(kernel=5, filters=20) 12x12x20 -> 8x8x20
        # max_pool(kernel=2) 8x8x20 -> 4x4x20
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))

        # flatten 4x4x20 = 320
        x = x.view(-1, 320)

        # 320 -> 50
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)

        # 50 -> 10
        x = self.fc2(x)

        # transform to logits
        return F.log_softmax(x)

class MNIST_ProblemClass:

    def __init__(self):
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data',
                                                                  download=True,
                                                                  train=True,
                                                                  transform=transforms.Compose([
                                                                      transforms.ToTensor(),
                                                                      # first, convert image to PyTorch tensor
                                                                      transforms.Normalize((0.1307,), (0.3081,))
                                                                      # normalize inputs
                                                                  ])),
                                                   batch_size=10,
                                                   shuffle=True)

        # download and transform test dataset
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data',
                                                                 download=True,
                                                                 train=False,
                                                                 transform=transforms.Compose([
                                                                     transforms.ToTensor(),
                                                                     # first, convert image to PyTorch tensor
                                                                     transforms.Normalize((0.1307,), (0.3081,))
                                                                     # normalize inputs
                                                                 ])),
                                                  batch_size=10,
                                                  shuffle=True)
        for batch_id, (data, label) in enumerate(train_loader):
            data = Variable(data)
            target = Variable(label)
        # self.model0 = MNIST_CNN()
        self.model0 = MNIST_MLP2()
        # print(self.model0)
        self.dataset = (data, target)
        # self.obj_function = self._obj_function(self.model0)
        self.obj_function = self._obj_function(self.model0)

    def _obj_function(self, model):
        x, y = self.dataset
        print(x.view(10, -1).shape)
        # y_hat = model(x).view(-1)
        y_hat = model(x.view(10, -1))
        # weight_norm = model[0].weight.norm() + model[2].weight.norm()
        # return F.binary_cross_entropy(y_hat, y) + 5e-4 / 2 * weight_norm
        return F.cross_entropy(y_hat, y)

    def get_model0(self):
        return self.model0

    def get_obj_function(self):
        return self.obj_function

    def get_dataset(self):
        return self.dataset


# prob = MNIST_ProblemClass()
# a = prob.get_obj_function()
# print(a)