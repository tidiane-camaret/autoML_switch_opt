import copy
import gym
import torch
from gym import spaces
import numpy as np
from torch import nn
import numpy as np
from problem import Variable
from omegaconf import OmegaConf
from modifedAdam import ModifiedAdam

from problem import Variable

device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'

config = OmegaConf.load('config.yaml')


def init_weights(m):
    # initialize weights of the model m
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
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
            problem_list,
            num_steps,
            history_len,
            optimization_mode='hard',
            lr=0.01,
            reward_system='function',
            threshold=0.01,
            adaptive_threshold=0.5,
            do_init_weights=True,
            optimizer_class_list=None,

    ):

        super().__init__()
        if optimizer_class_list is None:
            optimizer_class_list = [torch.optim.Adam, torch.optim.SGD, torch.optim.RMSprop]
        self.problem_list = problem_list  # list of problems
        self.num_steps = num_steps  # number of maximum steps per problem
        self.history_len = history_len  # number of previous steps to keep in the observation
        self.optimizer_class_list = optimizer_class_list
        self.do_init_weights = do_init_weights
        self.optimization_mode = optimization_mode
        self.lr = lr
        self._setup_episode()

        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.train_mode = True
        self.optimizer_storing_method = 'state_dict'
        self.reward_system = reward_system
        self.threshold = threshold
        self.adaptive_threshold = adaptive_threshold
        self.RT = 1
        self.PT = 10

        # Define action and observation space
        # Action space is the index of the optimizer class
        # that we want to use on the next step
        if self.optimization_mode == 'hard':
            self.action_space = spaces.Discrete(len(self.optimizer_class_list))
        elif self.optimization_mode == 'soft':
            self.action_space = spaces.Box(low=np.array([0.5, 0.998]), high=np.array([0.9, 0.999]), dtype=np.float32)
            # self.action_space = spaces.Discrete(4)
        else:
            print('mode of optimization is not set properly. Deufalut is hard')
            self.action_space = spaces.Discrete(len(self.optimizer_class_list))

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.history_len, 1 + self.num_params),
            dtype=np.float32,
        )

    # starting of a new episode
    def _setup_episode(self, problem=None):
        if problem is None:
            problem_index = np.random.randint(len(self.problem_list))  # randomly select a problem
            problem = self.problem_list[problem_index]

        self.model = copy.deepcopy(problem.model0)
        self.model.device = device
        self.objective_function = problem.obj_function
        if self.do_init_weights:
            self.model.apply(init_weights)

        if self.optimization_mode == 'hard':
            self.trained_optimizers = dict.fromkeys(self.optimizer_class_list)
            self.trained_optimizers_states = dict.fromkeys(self.optimizer_class_list)
            for key, _ in self.trained_optimizers.items():
                # initialise the optimisers
                optimizer_init = key(self.model.parameters(), lr=self.lr)
                self.trained_optimizers[key] = optimizer_init
                self.trained_optimizers_states[key] = optimizer_init.state_dict()

        else:

            optimizer_init = ModifiedAdam(self.model.parameters(),
                                          lr=self.lr)
            self.optimizer = optimizer_init

        self.obj_values = []
        self.gradients = []
        self.current_step = 0
        self.obj_values_sum = 0

    # reset the environment when the episode is over
    def reset(self, problem=None):
        self._setup_episode(problem)
        return make_observation(
            None, self.obj_values, self.gradients, self.num_params, self.history_len
        )


    # define the action : pick an optimizer
    # and update the model
    @torch.no_grad()
    def step(self, action):
        # here, an action is given by the agent
        # it is the index of the optimizer class
        # that we want to use on the next step
        # we calulate the new state and the reward
        if self.optimization_mode == 'soft':
            for param in self.optimizer.param_groups:
                param['betas'] = (action[0], action[1], 0.9, 0.999)  #(action[0] * 0.1 + 0.7, action[1] * 0.001 + 0.998, 0.9, 0.999)     #* 0.499 + 0.5
                # if action == 0:
                #     param['betas'] = (0.9, 0.999, 0.9, 0.999)
                # if action == 1:
                #     param['betas'] = (0.999, 0.9,  0.999, 0.9)
                # if action == 2:
                #     param['betas'] = (0.999, 0.999, 0.999, 0.999)
                # else:
                #     param['betas'] = (0.9, 0.9,0.9, 0.9)
            # update the model and
            # calculate the new objective value
            with torch.enable_grad():
                if isinstance(self.model, Variable):
                    traj_position = copy.deepcopy(self.model).x.detach().numpy()
                else:
                    traj_position = None
                obj_value = self.objective_function(self.model)
                self.optimizer.zero_grad()
                obj_value.backward()
                # update model parameters
                self.optimizer.step()
                # optimizer.zero_grad()
            # add the updated optimizer into list


        else:
            lookahead_obj_values = np.zeros(len(self.optimizer_class_list))
            # calculate the lookaead objective values (no optimizer update, this will be done in the next step)
            if self.reward_system == "lookahead" and self.train_mode:

                lookahead_steps = 2  # if we look only one step head, lookahead values are the same for all optimizers. TODO : See why.

                for o, opt_class in enumerate(self.optimizer_class_list):

                    model_decoy = copy.deepcopy(self.model)

                    current_optimizer = opt_class(model_decoy.parameters(), lr=self.lr)
                    current_optimizer.load_state_dict(self.trained_optimizers_states[opt_class])

                    for i in range(lookahead_steps):
                        obj_value = self.objective_function(model_decoy)
                        current_optimizer.zero_grad()
                        obj_value.backward()
                        current_optimizer.step()
                    lookahead_obj_values[o] = obj_value.item()
                # print("lookahead_obj_values", lookahead_obj_values)

            # update the parameters of every non chosen optimizer, on a decoy model (no update on the main model)

            for opt_class in self.optimizer_class_list:
                if opt_class != self.optimizer_class_list[action]:
                    model_decoy = copy.deepcopy(self.model)

                    current_optimizer = opt_class(model_decoy.parameters(), lr=self.lr)
                    current_optimizer.load_state_dict(self.trained_optimizers_states[opt_class])

                    with torch.enable_grad():
                        obj_value = self.objective_function(model_decoy)
                        current_optimizer.zero_grad()
                        obj_value.backward()
                        current_optimizer.step()
                    # add the updated optimizer into list

                    self.trained_optimizers_states[opt_class] = current_optimizer.state_dict()

            # load the chosen optimizer

            if self.optimizer_storing_method == "state_dict":
                optimizer_class = self.optimizer_class_list[action]
                optimizer = optimizer_class(self.model.parameters(), lr=self.lr)
                optimizer.load_state_dict(self.trained_optimizers_states[optimizer_class])

            elif self.optimizer_storing_method == "class_dict":
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
            if self.optimizer_storing_method == "state_dict":
                self.trained_optimizers_states[optimizer_class] = optimizer.state_dict()

            elif self.optimizer_storing_method == "class_dict":
                self.trained_optimizers[self.optimizer_class_list[action]] = optimizer

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

        # observation = np.ndarray.flatten(observation)

        obj_value = obj_value.item()
        self.obj_values_sum += obj_value

        if self.reward_system == "lookahead":
            reward = 1 if action == np.argmin(lookahead_obj_values) else -1
        elif self.reward_system == "threshold":
            reward = 10 if obj_value < self.threshold else -1
        elif self.reward_system == "inverse":
            reward = 1 / (obj_value + 1e-6)
        elif self.reward_system == "opposite":
            reward = -obj_value
        elif self.reward_system == 'adathreshold':
            if obj_value < self.threshold:
                reward = self.RT
                self.RT *= 1.01
                self.PT *= 0.99
                self.adaptive_threshold *= 0.99
            else:
                reward = self.PT
        else:
            raise NotImplementedError

        done = (self.current_step >= self.num_steps)

        info = {"obj_value": obj_value,
                "traj_position": traj_position}
        self.current_step += 1

        return observation, reward, done, info
