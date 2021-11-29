# AutoCkt: Deep Reinforcement Learning of Analog Circuit Designs
Code for [Deep Reinforcement Learning of Analog Circuit Designs](https://arxiv.org/abs/2001.01808), presented at Design Automation and Test in Europe, 2020. Note that the results shown in the paper include those from NGSpice and Spectre. NGSpice is free and can be installed online.
## Setup
This setup requires Anaconda. In order to obtain the required packages, run the command below from the top level directory of the repo to install the Anaconda environment:

```
conda env create -f environment.yml
```

To activate the environment run:
```
source activate autockt
```
You might need to install some packages further using pip if necessary. To ensure the right versions, look at the environment.yml file.

NGspice 2.7 needs to be installed separately, via this [installation link](https://sourceforge.net/projects/ngspice/files/ng-spice-rework/old-releases/27/). Page 607 of the pdf manual on the website has instructions on how to install. Note that you might need to remove some of the flags to get it to install correctly for your machine. 

## Training AutoCkt
Make sure that you are in the Anaconda environment. Before running training, the circuit netlist must be modified in order to point to the right library files in your directory. To do this, run the following command:
```
python eval_engines/ngspice/ngspice_inputs/correct_inputs.py 
```

To generate the design specifications that the agent trains on, run:
```
python autockt/gen_specs.py --num_specs ##
```
The result is a pickle file dumped to the gen_specs/ folder.

To train the agent, open ipython from the top level directory and then: 
```
run autockt/val_autobag_ray.py
```
The training checkpoints will be saved in your home directory under ray\_results. Tensorboard can be used to load reward and loss plots using the command:

```
tensorboard --logdir path/to/checkpoint
```

To replicate the results from the paper, num_specs 350 was used (only 50 were selected for each CPU worker). Ray parallelizes according to number of CPUs available, that affects training time. 
## Validating AutoCkt
The rollout script takes the trained agent and gives it new specs that the agent has never seen before. To generate new design specs, run the gen_specs.py file again with your desired number of specs to validate on. To run validation, open ipython:

```
run autockt/rollout.py /path/to/ray/checkpoint --run PPO --env opamp-v0 --num_val_specs ### --traj_len ## --no-render
``` 
* num_val_specs: the number of untrained objectives to test on
* traj_len: the length of each trajectory

Two pickle files will be updated: opamp_obs_reached_test and opamp_obs_nreached_test. These will contain all the reached and unreached specs, respectively.
