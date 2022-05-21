import os
import pickle
import random
from collections import OrderedDict

import gym
import numpy as np
import yaml
import yaml.constructor
from gym import spaces

from autockt.utils import OrderedDictYAMLLoader


class NgspiceEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config):
        self.multi_goal = env_config.get("multi_goal", True)
        self.generalize = env_config.get("generalize", True)
        num_valid = env_config.get("num_valid", 50)
        self.valid = env_config.get("run_valid", False)
        mode = "valid" if self.valid else "train"
        self.specs_path = (
            os.getcwd()
            + "/autockt/gen_specs/ngspice_specs_"
            + mode
            + "_"
            + env_config.get("env", None)
        )

        with open(self.CIR_YAML, "r") as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)

        # design specs
        if self.generalize == False:
            specs = yaml_data["target_specs"]
        else:
            with open(self.specs_path, "rb") as f:
                specs = pickle.load(f)

        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))

        self.specs_ideal = []
        self.specs_id = list(self.specs.keys())
        self.fixed_goal_idx = -1
        self.num_os = len(list(self.specs.values())[0])

        params = yaml_data["params"]
        self.params = []
        self.params_id = list(params.keys())

        for value in params.values():
            param_vec = np.arange(value[0], value[1], value[2])
            self.params.append(param_vec)

        # This should be overloaded in each env
        self.action_meaning = [-1, 0, 1]
        self.action_space = spaces.Tuple(
            [spaces.Discrete(len(self.action_meaning))] * len(self.params_id)
        )
        low_bound = np.array(
            [-np.inf] * 2 * len(self.specs_id) + [-np.inf] * len(self.params_id)
        )
        high_bound = np.array(
            [np.inf] * 2 * len(self.specs_id) + [np.inf] * len(self.params_id)
        )
        self.observation_space = spaces.Box(
            low=low_bound, high=high_bound, dtype=np.float32
        )

        # initialize current param/spec observations
        self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float32)
        self.cur_params = np.zeros(len(self.params_id), dtype=np.int32)

        # Get the g* (overall design spec) you want to reach
        self.global_g = []
        for spec in list(self.specs.values()):
            self.global_g.append(float(spec[self.fixed_goal_idx]))
        self.g_star = np.array(self.global_g)

        # objective number (used for validation)
        self.obj_idx = 0

    def reset(self):
        # if multi-goal is selected, every time reset occurs, it will select a different design spec as objective
        if self.generalize == True:
            if self.valid == True:
                if self.obj_idx > self.num_os - 1:
                    self.obj_idx = 0
                idx = self.obj_idx
                self.obj_idx += 1
            else:
                idx = random.randint(0, self.num_os - 1)
            self.specs_ideal = []
            for spec in list(self.specs.values()):
                self.specs_ideal.append(spec[idx])
            self.specs_ideal = np.array(self.specs_ideal)
        else:
            if self.multi_goal == False:
                self.specs_ideal = self.g_star
            else:
                idx = random.randint(0, self.num_os - 1)
                self.specs_ideal = []
                for spec in list(self.specs.values()):
                    self.specs_ideal.append(spec[idx])
                self.specs_ideal = np.array(self.specs_ideal)

        # initialize current parameters to
        self.cur_params = np.array([len(param_vec) // 2 for param_vec in self.params])
        self.cur_specs = self.update(self.cur_params)

        self.ob = np.concatenate([self.cur_specs, self.specs_ideal, self.cur_params])
        return self.ob

    def step(self, action):
        """
        :param action: is vector with elements between 0 and 1 mapped to the index of the corresponding parameter
        :return:
        """

        # Take action that RL agent returns to change current params
        action = list(np.reshape(np.array(action), (np.array(action).shape[0],)))
        self.cur_params = self.cur_params + np.array(
            [self.action_meaning[a] for a in action]
        )

        self.cur_params = np.clip(
            self.cur_params,
            [0] * len(self.params_id),
            [(len(param_vec) - 1) for param_vec in self.params],
        )
        # Get current specs and normalize
        self.cur_specs = self.update(self.cur_params)
        reward = self.reward(self.cur_specs, self.specs_ideal)
        done = False

        # incentivize reaching goal state
        if reward >= 10:
            done = True
            print("-" * 10)
            print("params = ", self.cur_params)
            print("specs:", self.cur_specs)
            print("ideal specs:", self.specs_ideal)
            print("-" * 10)

        self.ob = np.concatenate([self.cur_specs, self.specs_ideal, self.cur_params])
        return self.ob, reward, done, {}

    def lookup(self, spec, goal_spec):
        goal_spec = [float(e) for e in goal_spec]
        norm_spec = (spec - goal_spec) / (goal_spec + spec)
        return norm_spec

    def reward(self, spec, goal_spec):
        """
        Reward: doesn't penalize for overshooting spec, is negative
        """
        rel_specs = self.lookup(spec, goal_spec)
        reward = 0.0
        for i, rel_spec in enumerate(rel_specs):
            if self.specs_id[i] == "ibias_max":
                rel_spec = rel_spec * -1.0
            if rel_spec < 0:
                reward += rel_spec

        return reward if reward < -0.02 else 10

    def update(self, params_idx):
        """

        :param action: an int between 0 ... n-1
        :return:
        """
        params_idx = params_idx.astype(np.int32)
        params = [self.params[i][params_idx[i]] for i in range(len(self.params_id))]
        param_val = [OrderedDict(list(zip(self.params_id, params)))]

        # run param vals and simulate
        cur_specs = OrderedDict(
            sorted(
                self.sim_env.create_design_and_simulate(param_val[0])[1].items(),
                key=lambda k: k[0],
            )
        )
        cur_specs = np.array(list(cur_specs.values()))

        return cur_specs
