### define the environment
import gym 
import torch
from gym import spaces
import numpy as np

class Environment(gym.Env):
    def __init__(
        self,
        dataset,
        model,
        num_steps,
        history_len,
        optimizer_class_list = [torch.optim.SGD, torch.optim.Adam],

    ):

        super().__init__()

        self.dataset = dataset # list of problems 
        self.model = model # model to optimize
        self.num_steps = num_steps # number of maximum steps per problem
        self.history_len = history_len # number of previous steps to keep in the observation
        self.optimizer_class_list = optimizer_class_list

        self._setup_episode()
        self.num_params = sum(p.numel() for p in self.model.parameters())

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

        # starting of a new episode
        def _setup_episode(self):
            problem = np.random.choice(self.dataset)

            self.model.weight.data.random_(-1, 1)
            self.model.bias.data.random_(-1, 1)

            self.obj_values = []
            self.gradients = []
            self.current_step = 0

        # reset the environment when the episode is over
        def reset(self):
            self._setup_episode()
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

            # define the optimizer
            optimizer = self.optimizer_class_list[action](self.model.parameters())

        


