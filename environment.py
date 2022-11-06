### define the environment
import gym 
import torch
from gym import spaces
import numpy as np



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
        model,
        num_steps,
        history_len,
        objective_function,
        optimizer_class_list = [torch.optim.SGD, torch.optim.Adam],

    ):

        super().__init__()

        self.problem_list = problem_list # list of problems 
        self.model = model # model to optimize
        self.num_steps = num_steps # number of maximum steps per problem
        self.history_len = history_len # number of previous steps to keep in the observation
        self.optimizer_class_list = optimizer_class_list
        self.objective_function = objective_function
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
            problem = np.random.choice(self.problem_list)

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
            
            # update the model and 
            # calculate the new objective value
            optimizer.zero_grad() #do we need this ?
            obj_value = self.objective_function(self.model)
            obj_value.backward()
            optimizer.step()

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
            reward = -obj_value.item()
            done = self.current_step >= self.num_steps
            info = {}

            self.current_step += 1
            return observation, reward, done, info








