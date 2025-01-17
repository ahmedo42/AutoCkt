# AutoCkt: Deep Reinforcement Learning of Analog Circuit Designs
Code for [Deep Reinforcement Learning of Analog Circuit Designs](https://arxiv.org/abs/2001.01808).

## Note

- There are two branches, `master` and `valid-specs`, the primary difference is in the data generation script but both produce
    roughly the same generalization accuracy

- The `valid-specs` branch contain an almost-working implementation of the folded cascode circuit 

- This repo doesn't consistently produce the paper's results but can be a good starting point for analog design with RL

## Setup
Install Dependencies

```
pip install -r requirements.txt
```

Install Ngspice for simulation, On Ubuntu/Colab/Kaggle
```
sudo apt-get install -y ngspice
```


## Pseudocode of The training algorithm
<img src="./assets/pseudocode.png" width="500" height="400"></img>

## Training
Before running training, the circuit netlist must be modified in order to point to the right library files in your directory. To do this, run the following command:
```
python autockt/eval_engines/ngspice/ngspice_inputs/correct_inputs.py 
```

To generate the design specifications that the agent trains on, run:
```
python gen_specs.py --num_specs ## --env ## --mode ##
```
The result is a pickle file dumped to the gen_specs folder.

To train the agent: 
```
python train.py 
```

The training checkpoints will be saved under `results` in the main direcory. Tensorboard can be used to load reward and loss plots using the command:

```
tensorboard --logdir path/to/logs
```
## Validation
The rollout script takes the trained agent and gives it new specs that the agent has never seen before. To generate new design specs, run the gen_specs.py file again with your desired number of specs to validate on. To run validation:

```
python rollout.py /path/to/ray/checkpoint --run PPO --env opamp-v0 --num_val_specs ### --traj_len ## 
``` 
* num_val_specs: the number of untrained objectives to test on
* traj_len: the length of each trajectory

Two pickle files will be updated: opamp_obs_reached_test and opamp_obs_nreached_test. These will contain all the reached and unreached specs, respectively.
