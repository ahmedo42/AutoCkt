import os
import pickle
import random
from collections import OrderedDict

import gym
import numpy as np
import yaml
import yaml.constructor
from gym import spaces

from autockt.envs.ngspice_env import NgspiceEnv
from autockt.eval_engines.ngspice.TwoStageClass import *
from autockt.utils import OrderedDictYAMLLoader


class TwoStageAmp(NgspiceEnv):
    def __init__(self, env_config):
        self.CIR_YAML = (
            os.getcwd()
            + "/autockt/eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml"
        )
        self.sim_env = TwoStageClass(
            yaml_path=self.CIR_YAML, num_process=1, path=os.getcwd()
        )
        super().__init__(env_config)
