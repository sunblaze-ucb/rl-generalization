# Examples

## Running in a headless environment

In order to run these environments in a headless environment (e.g., without an X
server running), use `xvfb`:
```sh
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 -m examples.ppo2_baselines.train ...
```

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
```sh
python3 -m examples.epopt.train \
--env SunblazeCartPole-v0 \
--output epopt_cartpole \
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

To run experiments defined in `experiments.yml`:
```sh
python3 -m examples.run_experiments examples/experiments.yml /tmp/experiments-output
```
