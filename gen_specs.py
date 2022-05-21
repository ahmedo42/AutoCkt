import argparse
import pickle
import random

import numpy as np
import yaml

from autockt.utils import OrderedDictYAMLLoader


# Generate the design specifications and then save to a pickle file
def gen_data(CIR_YAML, env, num_specs, mode):
    with open(CIR_YAML, "r") as f:
        yaml_data = yaml.load(f, OrderedDictYAMLLoader)

    specs_range = yaml_data["target_specs"]
    specs_range_vals = list(specs_range.values())
    specs_valid = []
    for spec in specs_range_vals:
        if isinstance(spec[0], int):
            list_val = [
                random.randint(int(spec[0]), int(spec[1])) for x in range(0, num_specs)
            ]
        else:
            list_val = [
                random.uniform(float(spec[0]), float(spec[1]))
                for x in range(0, num_specs)
            ]
        specs_valid.append(tuple(list_val))
    i = 0
    for key, value in specs_range.items():
        specs_range[key] = specs_valid[i]
        i += 1
    with open("autockt/gen_specs/ngspice_specs_" + mode + "_" + env, "wb") as f:
        pickle.dump(specs_range, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_specs", type=int, default=50)
    parser.add_argument("--env", type=str, default="two_stage_opamp")
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    CIR_YAML = (
        "autockt/eval_engines/ngspice/ngspice_inputs/yaml_files/" + args.env + ".yaml"
    )

    gen_data(CIR_YAML, args.env, args.num_specs, args.mode)


if __name__ == "__main__":
    main()
