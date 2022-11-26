import copy
import gym
import torch
from gym import spaces
import numpy as np
from torch import nn
import numpy as np
from problem import Variable
from omegaconf import OmegaConf


config = OmegaConf.load('config.yaml')

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
        self.trained_optimizers_states = dict.fromkeys(self.optimizer_class_list)

        for key, _ in self.trained_optimizers.items():
            # initialise the optimisers
            optimizer_init = key(self.model.parameters(), lr=self.config.model.lr)
            self.trained_optimizers[key] = optimizer_init
            self.trained_optimizers_states[key] = optimizer_init.state_dict()

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


        # calculate the lookaead objective values (no optimizer update, this will be done in the next step)
        if config.environment.reward_system == "lookahead":
            lookahead_obj_values = []
            lookahead_steps = 2 # if we look only one step head, lookahead values are the same for all optimizers. TODO : See why.

            for opt_class in self.optimizer_class_list:

                model_decoy = copy.deepcopy(self.model)

                current_optimizer = opt_class(model_decoy.parameters(), lr=self.config.model.lr)
                current_optimizer.load_state_dict(self.trained_optimizers_states[opt_class]) 

                
                for i in range(lookahead_steps):
                    obj_value = self.objective_function(model_decoy)
                    current_optimizer.zero_grad()
                    obj_value.backward()
                    current_optimizer.step()
                lookahead_obj_values.append(obj_value.item())
            #print("lookahead_obj_values", lookahead_obj_values)

        # update the parameters of every non chosen optimizer, on a decoy model (no update on the main model)

        for opt_class in self.optimizer_class_list:
            if opt_class != self.optimizer_class_list[action]:

                model_decoy = copy.deepcopy(self.model)

                current_optimizer = opt_class(model_decoy.parameters(), lr=self.config.model.lr)
                current_optimizer.load_state_dict(self.trained_optimizers_states[opt_class])

                with torch.enable_grad():
                    obj_value = self.objective_function(model_decoy)
                    current_optimizer.zero_grad()
                    obj_value.backward()
                    current_optimizer.step()
            # add the updated optimizer into list

                self.trained_optimizers_states[opt_class] = current_optimizer.state_dict()

        # load the chosen optimizer 

        if config.environment.optimizer_storing_method == "state_dict":
            optimizer_class = self.optimizer_class_list[action]
            optimizer = optimizer_class(self.model.parameters(), lr=self.config.model.lr)
            optimizer.load_state_dict(self.trained_optimizers_states[optimizer_class])

        elif config.environment.optimizer_storing_method == "class_dict":
            optimizer = self.trained_optimizers[self.optimizer_class_list[action]]


        # update the model and the optimizer

        with torch.enable_grad():
            # if the problem is a low dimensional problem, extract the current point
            if isinstance(self.model, Variable):
                traj_position = copy.deepcopy(self.model).x.detach().numpy()
            else:
                traj_position = None
            obj_value = self.objective_function(self.model)
            optimizer.zero_grad()
            obj_value.backward()
            # update model parameters
            optimizer.step()

        # save the optimizer state
        if config.environment.optimizer_storing_method == "state_dict":
            self.trained_optimizers_states[optimizer_class] = optimizer.state_dict()

        elif config.environment.optimizer_storing_method == "class_dict":
            self.trained_optimizers[self.optimizer_class_list[action]] = optimizer


        # Calculate the current gradient and flatten it
        current_grad = torch.cat(
            [p.grad.flatten() for p in self.model.parameters()]
        ).flatten()

        obj_value = obj_value.item()

        # Update history of objective values and gradients with current objective
        # value and gradient.
        if len(self.obj_values) >= self.history_len:
            self.obj_values.pop(-1)
            self.gradients.pop(-1)
        self.obj_values.insert(0, obj_value)
        self.gradients.insert(0, current_grad)

        # Return observation, reward, done, and empty info
        observation = make_observation(
            obj_value,
            self.obj_values,
            self.gradients,
            self.num_params,
            self.history_len,
        )

        if config.environment.reward_system == "lookahead":
            reward = 1 if action == np.argmin(lookahead_obj_values) else 0
        elif config.environment.reward_system == "function":
            reward = self.reward_function(obj_value)
        else:
            raise NotImplementedError
        
        done = self.current_step >= self.num_steps

        info = {"obj_value" : obj_value,
                "traj_position" : traj_position}

        self.current_step += 1
        return observation, reward, done, info


