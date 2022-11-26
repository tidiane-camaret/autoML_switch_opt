import copy
import gym
import torch
from gym import spaces
import numpy as np
from torch import nn
from modifedAdam import ModifiedAdam


def init_weights(m):
    # initialize weights of the model m
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def make_observation(obj_value, obj_values, gradients, num_params, history_len):
    # Features is a matrix where the ith row is a concatenation of the difference
    # in the current objective value and that of the ith previous iterate as well
    # as the ith previous gradient.
    observation = np.zeros((history_len, 1 + num_params), dtype="float32")
    observation[: len(obj_values), 0] = (
            obj_value - torch.tensor(obj_values).detach().numpy()
    )
    for i, grad in enumerate(gradients):
        observation[i, 1:] = grad.detach().numpy()

    # Normalize and clip observation space
    observation /= 50
    return observation.clip(-1, 1)


class Environment(gym.Env):
    def __init__(
            self,
            config,
            problem_list,
            num_steps,
            history_len,
            #  optimizer_class_list,
            do_init_weights=True,
    ):

        super().__init__()
        self.config = config
        self.problem_list = problem_list  # list of problems
        self.num_steps = num_steps  # number of maximum steps per problem
        self.history_len = history_len  # number of previous steps to keep in the observation
        #   self.optimizer_class_list = optimizer_class_list
        self.do_init_weights = do_init_weights
        self._setup_episode()
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.optimizer = ModifiedAdam(self.model.parameters(), lr=config.model.lr)

        # Define action and observation space
        # Action space is the index of the optimizer class
        # that we want to use on the next step
        self.action_space = spaces.Box(low=np.array([0.01, 0.99]), high=np.array([0.999, 0.999]), dtype=np.float32)
        #spaces.Box(low=np.array([0.01, 0.01, 0.0, 0.0]), high=np.array([0.999, 0.999, 0.999, 0.999]), dtype=np.float32) #(action[0], 0.999, 0.0, 0.0)
        # spaces.Discrete(len(self.optimizer_class_list))
        # Observation space is the history of
        # the objective values and gradients

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.history_len, 1 + self.num_params),
            dtype=np.float32,
        )
        # create a numpy array of trained optimizers, initially filled with nans

    # starting of a new episode
    def _setup_episode(self, problem=None):
        if problem is None:
            problem_index = np.random.randint(len(self.problem_list))  # randomly select a problem
            problem = self.problem_list[problem_index]

        self.model = copy.deepcopy(problem.model0)
        self.objective_function = problem.obj_function
        if self.do_init_weights:
            self.model.apply(init_weights)
        # self.trained_optimizers = dict.fromkeys(self.optimizer_class_list)
        # for key, _ in self.trained_optimizers.items():
        #     # initialise the optimisers
        #     optimizer_init = key(self.model.parameters(), lr=self.config.model.lr)
        #     self.trained_optimizers[key] = optimizer_init

        self.obj_values = []
        self.gradients = []
        self.current_step = 0
        optimizer_init = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.model.lr)  # torch.optim.RMSprop(self.model.parameters(),lr=self.config.model.lr)
        self.optimizer = optimizer_init

    # reset the environment when the episode is over
    def reset(self, problem=None):
        self._setup_episode(problem)
        return make_observation(
            None, self.obj_values, self.gradients, self.num_params, self.history_len
        )

    # define the action : pick an optimizer
    # and update the model

    def step(self, action):
        print('Action is : ', action)
        # here, an action is given by the agent
        # it is the index of the optimizer class
        # that we want to use on the next step
        # we calulate the new state and the reward

        for param in self.optimizer.param_groups:
            param['betas'] = (action[0], action[1], 0, 0)  #

        # if action == 0:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.99, 0.999, 0.9, 0.999)  # (0.9, 0.999, 0.9, 0.999)
        # if action == 1:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.98, 0.999, 0.9, 0.999)
        # if action == 2:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.97, 0.999, 0.9, 0.999)
        # if action == 3:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.96, 0.999, 0.9, 0.999)
        # if action == 4:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.95, 0.999, 0.9, 0.999)
        # if action == 5:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.94, 0.999, 0.9, 0.999)
        # if action == 6:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.93, 0.999, 0.9, 0.999)
        # if action == 7:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.92, 0.999, 0.9, 0.999)
        # if action == 8:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.91, 0.999, 0.9, 0.999)
        # if action == 9:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.9, 0.999, 0.0, 0.999)

        # if action == 0:
        #     for param in self.optimizer.param_groups:
        #         param['alpha'] = (0.995)
        # if action == 1:
        #     for param in self.optimizer.param_groups:
        #         param['alpha'] = (0.990)
        # if action == 2:
        #     for param in self.optimizer.param_groups:
        #         param['alpha'] = (0.985)
        # if action == 3:
        #     for param in self.optimizer.param_groups:
        #         param['alpha'] = (0.980)
        # if action == 4:
        #     for param in self.optimizer.param_groups:
        #         param['alpha'] = (0.975)

        # if action == 0:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.99, 0.999, 0.01, 0.01)  # (0.9, 0.999, 0.9, 0.999)
        # if action == 1:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.99, 0.999, 0.02, 0.02)
        # if action == 2:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.99, 0.999, 0.03, 0.03)
        # if action == 3:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.99, 0.999, 0.04, 0.04)
        # if action == 4:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.99, 0.999, 0.05, 0.05)
        # if action == 5:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.99, 0.999, 0.06, 0.06)
        # if action == 6:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.99, 0.999, 0.07, 0.07)
        # if action == 7:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.99, 0.999, 0.08, 0.08)
        # if action == 8:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.99, 0.999, 0.09, 0.09)
        # if action == 9:
        #     for param in self.optimizer.param_groups:
        #         param['betas'] = (0.99, 0.999, 0.1, 0.1)

        # self.trained_optimizers[opt_class] = current_optimizer

        # use the optimizer that the agent selected to update model params
        #   optimizer_class = self.optimizer_class_list[action]
        #    optimizer = self.trained_optimizers[optimizer_class]

        # (self.model.parameters())

        # update the model and
        # calculate the new objective value
        with torch.enable_grad():
            obj_value = self.objective_function(self.model)
            self.optimizer.zero_grad()
            obj_value.backward()
            # update model parameters
            self.optimizer.step()
            # optimizer.zero_grad()
        # add the updated optimizer into list
        # self.trained_optimizers[optimizer_class] = optimizer.state_dict()

        # print(opt_s)

        # Calculate the current gradient and flatten it
        current_grad = torch.cat(
            [p.grad.flatten() for p in self.model.parameters()]
        ).flatten()

        # Update history of objective values and gradients with current objective
        # value and gradient.
        if len(self.obj_values) >= self.history_len:
            self.obj_values.pop(-1)
            self.gradients.pop(-1)
        self.obj_values.insert(0, obj_value)
        self.gradients.insert(0, current_grad)

        # Return observation, reward, done, and empty info
        observation = make_observation(
            obj_value.item(),
            self.obj_values,
            self.gradients,
            self.num_params,
            self.history_len,
        )
        observation.flatten()
        reward = -obj_value.item()
        done = self.current_step >= self.num_steps
        info = {}

        self.current_step += 1
        return observation, reward, done, info


def eval_handcrafted_optimizer(problem_list, optimizer_class, num_steps, config, do_init_weights=False):
    """
    Run an optimizer on a list of problems
    """
    rewards = []
    for problem in problem_list:
        model = copy.deepcopy(problem.model0)
        if do_init_weights:
            model.apply(init_weights)

        optimizer = optimizer_class(model.parameters(), lr=config.model.lr)
        obj_values = []
        for step in range(num_steps):
            obj_value = problem.obj_function(model)
            obj_values.append(-obj_value.detach().numpy())
            optimizer.zero_grad()
            obj_value.backward()
            optimizer.step()
        rewards.append(obj_values)
    return np.array(rewards)
