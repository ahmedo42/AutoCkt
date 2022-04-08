import os
import pickle
import random
from collections import OrderedDict

import gym
import numpy as np
import yaml
import yaml.constructor
from autockt.eval_engines.ngspice.csamp import *
from autockt.utils import OrderedDictYAMLLoader
from gym import spaces
from scipy import spatial 

class FoldedCascode(NgspiceEnv):
    def __init__(self, env_config):
        self.df = pd.read_csv("autockt/eval_engines/ddb/folded_cascode.csv")
        self.CIR_YAML = os.getcwd()+'/autockt/eval_engines/ddb/folded_cascode.yaml'
        super().__init__(env_config)

    def _get_closest_spec(self,params):
        if not hasattr(self, "params_tree"):
            params_df = self.df[self.params_id]
            params_df = params_df.reindex(sorted(params_df.columns), axis=1)
            self.params_tree = spatial.KDTree(params_df.values) 
        
        idx = self.params_tree.query(params)[1]
        result = self.df.iloc[[idx]]
        result = result[self.specs_id]
        result = result.reindex(sorted(result.columns), axis=1)
        return result.values[0]

    def update(self,params_idx):
        params_idx = params_idx.astype(np.int32)
        params = [self.params[i][params_idx[i]] for i in range(len(self.params_id))]
        cur_specs = self._get_closest_spec(params)
        return cur_specs