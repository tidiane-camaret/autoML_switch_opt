from scripts.main import train_and_eval_agent
import wandb
from problem import NoisyHillsProblem, GaussianHillsProblem, MNISTProblemClass, ImageDatasetProblemClass
import numpy as np
import torch
from omegaconf import OmegaConf
import torchvision.datasets as datasets
from eval_functions import eval_agent, eval_handcrafted_optimizer
config = OmegaConf.load("config.yaml")
### parameters specific to math problems
math_problem_train_class = GaussianHillsProblem
math_problem_eval_class = GaussianHillsProblem
xlim = 2
nb_train_points = 1000
nb_test_points = 500
binary_classes = [[2,3], [4,5], [6,7], [8,9]]
dataset_class = datasets.KMNIST
problem_list = [ImageDatasetProblemClass(classes = bc, dataset_class = dataset_class) for bc in binary_classes]
optimizer_class_list = [torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop]
model_training_steps = config.model.model_training_steps

sweep_config = {
    "method": "grid",
    "parameters": {
                    "lr": {
                        "values": [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
                        },
                    }
}

sweep_id = wandb.sweep(sweep_config, project="mnist_lr_tuning")

def sweep_function():
        run = wandb.init()
        for optimizer_class in optimizer_class_list:
            obj_values, trajectories = eval_handcrafted_optimizer(problem_list, 
                                                            optimizer_class, 
                                                            model_training_steps, 
                                                            config, 
                                                            do_init_weights=False,
                                                            lr=wandb.config.lr)
            wandb.log({"{}_obj_values".format(optimizer_class.__name__): wandb.Histogram(np.array(obj_values))})
            wandb.log({"{}_mean_obj_value".format(optimizer_class.__name__): np.mean(obj_values)})

wandb.agent(sweep_id, function=sweep_function)

