## Q-learning on DualTaxi environment

This repository contains code for Q-learning on a forked version of OpenAI gym's Taxi-v3 environment. Check the file `gym/gym/envs/toy_text/dual_taxi.py` for detailed documentation about how the environment works. 

## Running the code

You need to install jupypter (or jupyter-lab) on your system. Both can be installed through Python's package manager pip.

The dependencies can be installed using the requirements.txt file (there's a command for the same in the notebook too)

```sh
pip install -r requirements.txt
```

Note: The environment `DualTaxi-v1` is a custom environment and not a part of official gym repository, so make sure you're installing local copy provided in this repository and not the official one (this is already taken care of in the requirements.txt file of this repository).

If you are able to run the following snippet correctly, it means everything is setup just fine :)

```
import gym
env = gym.make('DualTaxi-v1')
env.render()
```

All the q-learning code is inside the jupyter notebook provided in the repository. 

