import numpy as np
from problem import Variable
import copy
from environment import init_weights

def eval_agent(env, policy, problem_list = None, num_steps=5, random_actions=False):

    if problem_list is None:
        problem_list = env.problem_list

    actions, obj_values = np.zeros((len(problem_list), num_steps)), np.zeros((len(problem_list), num_steps))

    for episode, problem in enumerate(problem_list):
        obs = env.reset(problem=problem)
        for step in range(num_steps):
            if random_actions:
                action = env.action_space.sample()
            else:
                action, _states = policy.predict(obs)
            obs, reward, done, info = env.step(action)
            actions[episode, step] = action
            obj_values[episode, step] = info["obj_value"]
            if done:
                break
    return actions, obj_values


def eval_handcrafted_optimizer(problem_list, optimizer_class, num_steps, config, do_init_weights=False):
    """
    Run an optimizer on a list of problems
    """
    obj_values = []
    trajectories = []
    for problem in problem_list:
        model = copy.deepcopy(problem.model0)
        if do_init_weights:
            model.apply(init_weights)

        optimizer = optimizer_class(model.parameters(), lr=config.model.lr)
        o_v = []
        t = []
        for step in range(num_steps):
            # if model is a Variable instance, extract the parameter values
            if isinstance(model, Variable):
                t.append(copy.deepcopy(model).x.detach().numpy())
            obj_value = problem.obj_function(model)
            o_v.append(obj_value.detach().numpy())
            optimizer.zero_grad()
            obj_value.backward()
            optimizer.step()
        obj_values.append(o_v)
        trajectories.append(t)
    return np.array(obj_values), trajectories

def eval_switcher_optimizer(problem_list, optimizer_class_list, num_steps, config, switch_time=0.5, do_init_weights=False):
    """
    Run an optimizer on a list of problems
    """
    rewards = []
    for problem in problem_list:
        model = copy.deepcopy(problem.model0)
        if do_init_weights:
            model.apply(init_weights)

        optimizer = optimizer_class_list[0](model.parameters(), lr=config.model.lr)
        obj_values = []
        for step in range(int(num_steps*switch_time)):
            obj_value = problem.obj_function(model)
            obj_values.append(obj_value.detach().numpy())
            optimizer.zero_grad()
            obj_value.backward()
            optimizer.step()

        optimizer = optimizer_class_list[1](model.parameters(), lr=config.model.lr)
        for step in range(int(num_steps*switch_time), num_steps):
            obj_value = problem.obj_function(model)
            obj_values.append(obj_value.detach().numpy())
            optimizer.zero_grad()
            obj_value.backward()
            optimizer.step()
        rewards.append(obj_values)
    return np.array(rewards)