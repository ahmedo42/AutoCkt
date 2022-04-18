import os
import pickle
import random
from collections import OrderedDict

import yaml
import numpy as np
import pandas as pd
from autockt.utils import OrderedDictYAMLLoader
from scipy import spatial 

class FoldedCascode:
    def __init__(self):
        self.df = pd.read_csv("autockt/eval_engines/ddb/folded_cascode.csv")
        self.CIR_YAML = os.getcwd()+'/autockt/eval_engines/ddb/folded_cascode.yaml'
        with open(self.CIR_YAML, 'r') as f:
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)
        self.specs_id = sorted(list(yaml_data['target_specs'].keys()))
        self._init_params(yaml_data['params'])

    def _init_params(self,params_dict):
        self.params = []
        self.params_id = list(params_dict.keys())
        for value in params_dict.values():
            param_vec = np.arange(value[0], value[1], value[2]).tolist()
            self.params.append(param_vec)

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