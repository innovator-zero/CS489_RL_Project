# Introduction

This project contains three algorithms, DQN, PPO, and SAC.

For DQN, supported Atari environments:

- PongNoFrameSkip-v4
- BoxingNoFrameSkip-v4

For PPO and SAC, supported MuJoCo environments:

- Ant-v2
- Hopper-v2
- HalfCheetah-v2

# Usage

You can run ```run.py``` like 

```python run.py --env_name ENV_NAME --method METHOD (--train)```

where you can assign the environment to run by ```--env_name```,  and assign the algorithm to use in a MuJoCo environment by ```--method```, like ```--method PPO``` or ```--method SAC``` (default).

The default use is to load the trained models and test the algorithms in environments,  it will run 100 episodes and the mean and variance of the scores will be printed.

If you want to train a model by yourself, you can add ```--train```.

