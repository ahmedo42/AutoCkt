import argparse
import pickle
import random
from collections import OrderedDict

import numpy as np
import yaml

from autockt.utils import OrderedDictYAMLLoader
from autockt.eval_engines.ngspice.TwoStageClass import *
from autockt.eval_engines.ngspice.csamp import *

#Generate the design specifications and then save to a pickle file
def gen_data(CIR_YAML, env, num_specs, mode, sim_env):
  with open(CIR_YAML, 'r') as f:
    yaml_data = yaml.load(f, OrderedDictYAMLLoader)

  yaml_specs = yaml_data['target_specs']
  yaml_params= yaml_data['params']
  param_vals = list(yaml_params.values())
  spec_ranges = list(yaml_specs.values())

  params = []
  valid_specs = []
  params_id = list(yaml_params.keys())

  for value in yaml_params.values():
      param_vec = np.arange(value[0], value[1], value[2])
      params.append(param_vec)

  while len(valid_specs) < num_specs:
    random_params = np.array([random.randint(0, len(param_vec)-1) for param_vec in params])
    specs = simulate(random_params,params,sim_env,params_id)
    valid_specs.append(tuple(specs))
  for idx,(key,value) in enumerate(sorted(yaml_specs.items(),key=lambda k:k[0])):
      yaml_specs[key] = tuple([valid_specs[i][idx] for i in range(len(valid_specs))])
  with open("autockt/gen_specs/ngspice_specs_"+mode+'_'+env, 'wb') as f:
    pickle.dump(yaml_specs,f)

def simulate(params_idx,params,sim_env,params_id):

    params_idx = params_idx.astype(np.int32)
    new_params = [params[i][params_idx[i]] for i in range(len(params_id))]
    param_val = [OrderedDict(list(zip(params_id,new_params)))]
    
    specs = OrderedDict(sorted(sim_env.create_design_and_simulate(param_val[0])[1].items(), key=lambda k:k[0]))
    specs = np.array(list(specs.values()))

    return specs


def is_spec_valid(spec,spec_ranges):
  return all([spec[i] >= spec_ranges[i][0] and spec[i] <= spec_ranges[i][1] for i in range(len(spec_ranges))])

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_specs", type=int,default=50)
  parser.add_argument("--env", type=str, default="two_stage_opamp")
  parser.add_argument("--mode", type=str, default="train")
  args = parser.parse_args()
  CIR_YAML = "autockt/eval_engines/ngspice/ngspice_inputs/yaml_files/" + args.env + ".yaml"
  sim_env = TwoStageClass(yaml_path=CIR_YAML, num_process=1, path=os.getcwd()) 
  gen_data(CIR_YAML, args.env, args.num_specs, args.mode, sim_env)

if __name__=="__main__":
  main()
