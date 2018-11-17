# Examples

Please refer to the README in the `sunblaze_envs` folder for descriptions of the environments.

## Policy and value function architectures

We consider two architectures for the policy and value function:
* **mlp**: Policy and value function are MLPs with two hidden layers and no parameter sharing.
* **lstm**: Policy and value function are separate fully connected layers on top of a LSTM whose inputs are learned features computed using a MLP.

Please refer to the paper for details.

## PPO

To train with OpenAI Baselines PPO2:
```sh
python3 -m examples.ppo2_baselines.train \
--env SunblazeCartPole-v0 \
--output ppo2_cartpole \
--policy mlp \
--total-episodes 10000
```

## A2C

To train with OpenAI Baselines A2C:
```sh
python3 -m examples.a2c_baselines.train \
--env SunblazeCartPole-v0 \
--output a2c_cartpole \
--policy lstm \
--total-episodes 10000
```

## EPOpt

To train with EPOpt (based on the OpenAI Baselines PPO/A2C code):

Under the **mlp** policy:
```sh
python3 -m examples.epopt.train \
--env SunblazeCartPole-v0 \
--output epopt_cartpole \
--algorithm ppo2 \
--total-episodes 10000
```

Under the **lstm** policy:
```sh
python3 -m examples.epopt_lstm.train \
--env SunblazeCartPole-v0 \
--output epopt_lstm_cartpole \
--algorithm ppo2 \
--total-episodes 10000
```

## RL<sup>2</sup>

To train with RL<sup>2</sup> (also based on the OpenAI Baselines PPO/A2C code):
```sh
python3 -m examples.adaptive.train \
--env SunblazeCartPole-v0 \
--output rl2_cartpole \
--algorithm ppo2 \
--trials 5000 \
--episodes-per-trial 2
```

## Experiment Runner

To run one set of the experiments in the paper, using `experiments.yml`:
```sh
python3 -m examples.run_experiments examples/experiments.yml /tmp/experiments-output
```

## Running in a headless environment

In order to run these environments in a headless environment (e.g., without an X
server running), use `xvfb`:
```sh
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 -m examples.ppo2_baselines.train ...
```
