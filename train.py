import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_workers",type=int,default=1)
parser.add_argument("--horizon",type=int,default=30)
parser.add_argument("--batch_size",type=int,default=1200)
parser.add_argument("--framework",type=str,default="torch")
parser.add_argument("--algo",type=str,default="PPO")
args = parser.parse_args()
ray.init()

#configures training of the agent with associated hyperparameters
#See Ray documentation for details on each parameter
config_train = {
            "train_batch_size": args.batch_size,
            "horizon":  args.horizon,
            "model":{"fcnet_hiddens": [64, 64]},
            "num_workers": args.num_workers,
            "env_config":{"generalize":True, "run_valid":False},
            "framework":args.framework,
            }

#Runs training and saves the result in ~/ray_results/train_ngspice_45nm
#If checkpoint fails for any reason, training can be restored 
trials = tune.run_experiments({
    "train_45nm_ngspice": {
    "checkpoint_freq":1,
    "run": args.algo,
    "env": TwoStageAmp,
    "stop": {"episode_reward_mean": -0.02},
    "checkpoint_at_end":True,
    "config": config_train},
})
