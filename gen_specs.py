import argparse
import pickle
import random
from collections import OrderedDict

import numpy as np
import yaml
import pandas as pd

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
  sorted_specs = sorted(yaml_specs.items(),key=lambda k:k[0])
  specs_ranges = [y for x,y in sorted_specs]

  valid_specs = []

  if isinstance(sim_env, pd.DataFrame):
    spec_names = [x for x,y in sorted_specs]
    df = sim_env.sample(n=num_specs)
    print(spec_names)
    df = df[spec_names]
    df = df.reindex(sorted(df.columns), axis=1)
    specs = df.values
    for spec in specs:
      print(spec)
      valid_specs.append(tuple(spec))

  else:
    params = []
    params_id = list(yaml_params.keys())

    for value in yaml_params.values():
        param_vec = np.arange(value[0], value[1], value[2])
        params.append(param_vec)

    while len(valid_specs) < num_specs:
      random_params = np.array([random.randint(0, len(param_vec)-1) for param_vec in params])
      specs = simulate(random_params,params,sim_env,params_id)

      if is_spec_valid(specs,specs_ranges):
        print(specs)
        valid_specs.append(tuple(specs))

  for idx,(key,value) in enumerate(sorted_specs):
      yaml_specs[key] = tuple([valid_specs[i][idx] for i in range(len(valid_specs))])
  with open("autockt/gen_specs/specs_"+mode+'_'+env, 'wb') as f:
    pickle.dump(yaml_specs,f)

def simulate(params_idx,params,sim_env,params_id):

    params_idx = params_idx.astype(np.int32)
    new_params = [params[i][params_idx[i]] for i in range(len(params_id))]
    param_val = [OrderedDict(list(zip(params_id,new_params)))]
    
    specs = OrderedDict(sorted(sim_env.create_design_and_simulate(param_val[0])[1].items(), key=lambda k:k[0]))
    specs = np.array(list(specs.values()))

    return specs


def is_spec_valid(specs,specs_ranges):
    for spec,(lower_bound,upper_bound) in zip(specs,specs_ranges):
      if spec < lower_bound:
        return False
    return True

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_specs", type=int,default=50)
  parser.add_argument("--env", type=str, default="two_stage_opamp")
  parser.add_argument("--mode", type=str, default="train")
  args = parser.parse_args()
  CIR_YAML = "autockt/eval_engines/ngspice/ngspice_inputs/yaml_files/" + args.env + ".yaml"
  DATAFRAME_PATH = "autockt/eval_engines/ddb/" + args.env + ".csv"
  if args.env == "two_stage_opamp":
    sim_env = TwoStageClass(yaml_path=CIR_YAML, num_process=1, path=os.getcwd()) 
  elif args.env == "cs_amp":
    sim_env = CsAmpClass(yaml_path=CIR_YAML, num_process=1, path=os.getcwd())
  elif args.env == "folded_cascode":
    sim_env = pd.read_csv(DATAFRAME_PATH)
    CIR_YAML = "autockt/eval_engines/ddb/" + args.env + ".yaml"

  gen_data(CIR_YAML, args.env, args.num_specs, args.mode, sim_env)

if __name__=="__main__":
  main()
