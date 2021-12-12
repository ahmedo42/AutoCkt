import argparse
import pickle
import random
from collections import OrderedDict

import numpy as np
import yaml

from autockt.utils import OrderedDictYAMLLoader


#Generate the design specifications and then save to a pickle file
def gen_data(CIR_YAML, env, num_specs):
  with open(CIR_YAML, 'r') as f:
    yaml_data = yaml.load(f, OrderedDictYAMLLoader)

  specs_range = yaml_data['target_specs']
  specs_range_vals = list(specs_range.values())
  specs_valid = []
  for spec in specs_range_vals:
      if isinstance(spec[0],int):
          list_val = [random.randint(int(spec[0]),int(spec[1])) for x in range(0,num_specs)]
      else:
          list_val = [random.uniform(float(spec[0]),float(spec[1])) for x in range(0,num_specs)]
      specs_valid.append(tuple(list_val))
  i=0
  for key,value in specs_range.items():
      specs_range[key] = specs_valid[i]
      i+=1
  with open("autockt/gen_specs/ngspice_specs_gen_"+env, 'wb') as f:
    pickle.dump(specs_range,f)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--num_specs", type=int,default=100)
  args = parser.parse_args()
  CIR_YAML = "autockt/eval_engines/ngspice/ngspice_inputs/yaml_files/two_stage_opamp.yaml"
  
  gen_data(CIR_YAML, "two_stage_opamp", args.num_specs)

if __name__=="__main__":
  main()
