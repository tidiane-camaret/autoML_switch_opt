import torch.nn as nn
import torch


def buildmodel():
    # initialize weights of the model m

    model = nn.Sequential(
        nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1), nn.Sigmoid()
    )

    torch.nn.init.xavier_uniform_(model.weight)
    model.bias.data.fill_(0.01)

    return model
