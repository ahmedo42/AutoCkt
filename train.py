import ray
import ray.tune as tune
from ray.tune import CLIReporter
from autockt.envs.ngspice_vanilla_opamp import TwoStageAmp
from autockt.envs.ngspice_csamp import CsAmp
from autockt.envs.folded_cascode import FoldedCascode

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
parser.add_argument("--seed",type=int,default=17)
parser.add_argument("--lr",type=float,default=5e-5)
parser.add_argument("--neurons",type=int,default=64)
parser.add_argument("--n_layers",type=int,default=2)
parser.add_argument("--mid_range_init",action="store_true")
args = parser.parse_args()
ray.init()

#configures training of the agent with associated hyperparameters
#See Ray documentation for details on each parameter

env_mapping = {
    "two_stage_opamp" : TwoStageAmp,
    "cs_amp": CsAmp,
    "folded_cascode":FoldedCascode,
}

model_structure = [args.neurons] * args.n_layers
config_train = {
            "train_batch_size": args.batch_size,
            "horizon":  args.horizon,
            "model":{"fcnet_hiddens": model_structure},
            "num_workers": args.num_workers,
            "env_config":{"generalize":True, "run_valid":False,"env":args.env,"mid_range_init":args.mid_range_init},
            "framework":args.framework,
            "episodes_per_batch":args.episodes_per_batch,
            "seed" : args.seed,
            "lr" : args.lr,

            }

if args.algo != "ES":
    del config_train["episodes_per_batch"]

reporter = CLIReporter(max_report_frequency=args.log_frequency)
env = env_mapping[args.env]

#Runs training and saves the result in ~/ray_results/{args.experiment}
#If checkpoint fails for any reason, training can be restored 
trials = tune.run_experiments({
    args.experiment: {
    "run": args.algo,
    "checkpoint_freq":1,
    "keep_checkpoints_num":1,
    "local_dir":args.output_path,
    "env": env,
    "stop": {"episode_reward_mean": -0.02},
    "checkpoint_at_end":True,
    "config": config_train}},
    progress_reporter=reporter
)
