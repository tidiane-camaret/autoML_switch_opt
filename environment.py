import copy
import gym
import torch
from gym import spaces
import numpy as np
from torch import nn


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
            optimizer_class_list,
            do_init_weights=True,
            reward_function = lambda x: -x,

    ):

        super().__init__()
        self.config = config
        self.problem_list = problem_list  # list of problems
        self.num_steps = num_steps  # number of maximum steps per problem
        self.history_len = history_len  # number of previous steps to keep in the observation
        self.optimizer_class_list = optimizer_class_list
        self.do_init_weights = do_init_weights
        self._setup_episode()
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.reward_function = reward_function

        # Define action and observation space
        # Action space is the index of the optimizer class
        # that we want to use on the next step
        self.action_space = spaces.Discrete(len(self.optimizer_class_list))

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
            problem_index = np.random.randint(len(self.problem_list)) # randomly select a problem
            problem = self.problem_list[problem_index]


        self.model = copy.deepcopy(problem.model0)
        self.objective_function = problem.obj_function
        if self.do_init_weights:
            self.model.apply(init_weights)
        self.trained_optimizers = dict.fromkeys(self.optimizer_class_list)
        for key, _ in self.trained_optimizers.items():
            # initialise the optimisers
            optimizer_init = key(self.model.parameters(), lr=self.config.model.lr)
            self.trained_optimizers[key] = optimizer_init

        self.obj_values = []
        self.gradients = []
        self.current_step = 0

    # reset the environment when the episode is over
    def reset(self, problem=None):
        self._setup_episode(problem)
        return make_observation(
            None, self.obj_values, self.gradients, self.num_params, self.history_len
        )

    # define the action : pick an optimizer
    # and update the model

    def step(self, action):
        # here, an action is given by the agent
        # it is the index of the optimizer class
        # that we want to use on the next step
        # we calulate the new state and the reward

        # update the parameters of all optimizers,
        # this is to take care of information passing across optimizers of different classes

        for opt_class in self.optimizer_class_list:
            # calculate the gradients for all optimizers
            current_optimizer = self.trained_optimizers[opt_class]
            # optimizer.load_state_dict(self.trained_optimizers[opt_class]) #do we need this?
            with torch.enable_grad():
                obj_value = self.objective_function(self.model)
                current_optimizer.zero_grad()
                obj_value.backward()
            # add the updated optimizer into list
            self.trained_optimizers[opt_class] = current_optimizer

        # use the optimizer that the agent selected to update model params
        optimizer_class = self.optimizer_class_list[action]
        optimizer = self.trained_optimizers[optimizer_class]

        # (self.model.parameters())

        # update the model and
        # calculate the new objective value
        with torch.enable_grad():
            obj_value = self.objective_function(self.model)
            optimizer.zero_grad()
            obj_value.backward()
            # update model parameters
            optimizer.step()
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

        obj_value = obj_value.item()
        reward = self.reward_function(obj_value)
        done = self.current_step >= self.num_steps
        info = {"obj_value" : obj_value}

        self.current_step += 1
        return observation, reward, done, info


