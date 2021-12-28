import ray
import ray.tune as tune
from ray.tune import CLIReporter
from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp
from autockt.envs.ngspice_csamp import CsAmp

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_workers",type=int,default=1)
parser.add_argument("--experiment",type=str,default="train_45nm_ngspice")
parser.add_argument("--horizon",type=int,default=30)
parser.add_argument("--batch_size",type=int,default=1200)
parser.add_argument("--framework",type=str,default="torch")
parser.add_argument("--algo",type=str,default="PPO")
parser.add_argument("--episodes_per_batch",type=int,default=100) # valid only when training with ES
parser.add_argument("--output_path",type=str,default="./results")
parser.add_argument("--log_frequency",type=int,default=30) # number of seconds between each log
parser.add_argument("--env",type=str,default="two_stage_opamp")
args = parser.parse_args()
ray.init()

#configures training of the agent with associated hyperparameters
#See Ray documentation for details on each parameter

env_mapping = {
    "two_stage_opamp" : TwoStageAmp,
    "cs_amp": CsAmp,
}

config_train = {
            "train_batch_size": args.batch_size,
            "horizon":  args.horizon,
            "model":{"fcnet_hiddens": [50, 50, 50]},
            "num_workers": args.num_workers,
            "env_config":{"generalize":True, "run_valid":False,"env":args.env},
            "framework":args.framework,
            "episodes_per_batch":args.episodes_per_batch,
            }

if args.algo != "ES":
    del config_train["episodes_per_batch"]

reporter = CLIReporter(max_report_frequency=args.log_frequency)
env = env_mapping[args.env]

#Runs training and saves the result in ~/ray_results/train_ngspice_45nm
#If checkpoint fails for any reason, training can be restored 
trials = tune.run_experiments({
    args.experiment: {
    "run": args.algo,
    "checkpoint_freq":5,
    "keep_checkpoints_num":1,
    "local_dir":args.output_path,
    "env": env,
    "stop": {"episode_reward_mean": -0.02},
    "checkpoint_at_end":True,
    "config": config_train}},
    progress_reporter=reporter
)
