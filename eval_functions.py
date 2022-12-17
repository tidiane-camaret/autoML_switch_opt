import numpy as np
from problem import Variable
import copy
from environment import init_weights
from omegaconf import OmegaConf

config = OmegaConf.load('config.yaml')

def eval_agent(env, policy, problem_list=None,  num_episodes=10, num_steps=5, random_actions=False):
    if True:
        if problem_list is None:
            problem_list = env.problem_list

        actions = []
        trajectories = []
        obj_values = np.zeros((len(problem_list), num_steps))

        for episode, problem in enumerate(problem_list):
            print('episode ', episode, )
            t = []
            a = []
            obs = env.reset(problem=problem)
            for step in range(num_steps):
                if random_actions:
                    action = env.action_space.sample()
                else:
                    action, _states = policy.predict(obs)
                obs, reward, done, info = env.step(action)
                obj_values[episode, step] = info["obj_value"] #info["obj_value_test"]
                a.append(action)
                t.append(info["traj_position"])
                if done:
                    break
            trajectories.append(t)
            actions.append(a)
        return obj_values, np.array(trajectories), np.array(actions)




def eval_handcrafted_optimizer(problem_list, optimizer_class, num_steps, config, do_init_weights=False, optimizer_2_class=None, switch_time=None, lr = config.model.lr):
    """
    Run an optimizer on a list of problems
    """
    obj_values = []
    trajectories = []
    for problem in problem_list:
        model = copy.deepcopy(problem.model0)
        if do_init_weights:
            model.apply(init_weights)
        o_v = []
        t = []
        optimizer = optimizer_class(model.parameters(), lr=lr)
        for step in range(num_steps):
            if optimizer_2_class is not None and step == int(num_steps*switch_time):
                optimizer = optimizer_2_class(model.parameters(), lr=config.model.lr)
            # if model is a Variable instance, extract the parameter values
            if isinstance(model, Variable):
                t.append(copy.deepcopy(model).x.detach().numpy())
            obj_value = problem.obj_function(model)
            o_v.append(obj_value.detach().numpy())#problem.obj_function(model, mode = "test").detach().numpy())
            optimizer.zero_grad()
            obj_value.backward()
            optimizer.step()
        obj_values.append(o_v)
        trajectories.append(t)
    return np.array(obj_values), np.array(trajectories)


def first_index_below_threshold(array, threshold):
    """Return the first index of an array below a threshold. if none, return last index."""
    for i, x in enumerate(array):
        if x < threshold:
            return i
    return len(array)-1