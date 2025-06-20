# Where to Intervene: Action Selection in Deep Reinforcement Learning

This repository contains the code for Where to Intervene: Action Selection in Deep Reinforcement Learning.

## Getting Started

Firstly, you need to install Mujoco (https://github.com/openai/mujoco-py)

```sh
conda create --name rl python=3.9 -y

source activate rl
```

To install the dependencies, run the following command

```sh
cd RL_High_action
# Install all python dependencies
chmod +x install.sh
source install.sh
```

## Experiments

In Method/run_rl.py, Update wandb key (line 45) and project name (line 71)

Then you can run:

```sh
bash scripts/run_ppo.sh
```
