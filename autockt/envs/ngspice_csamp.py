import os
import pickle
import random
from collections import OrderedDict

import gym
import numpy as np
import yaml
import yaml.constructor
from autockt.eval_engines.ngspice.csamp import *
from autockt.envs.ngspice_env import  NgspiceEnv
from autockt.utils import OrderedDictYAMLLoader
from gym import spaces

class CsAmp(NgspiceEnv):
    def __init__(self, env_config):
        self.CIR_YAML = os.getcwd()+'/autockt/eval_engines/ngspice/ngspice_inputs/yaml_files/cs_amp.yaml'
        self.sim_env = CsAmpClass(yaml_path=self.CIR_YAML, num_process=1, path=os.getcwd()) 
        super().__init__(env_config)

