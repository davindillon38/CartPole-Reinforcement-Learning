
# wandb
The wandb logs for my runs are [here](https://wandb.ai/davindillon-ohio-university/cartpole-qlearning?nw=nwuserdavindillon)


# CartPole-Reinforcement-Learning
Reinforcement learning approach to OpenAI Gym's CartPole environment


## Description
The cartpole problem is an inverted pendelum problem where a stick is balanced upright on a cart. The cart can be moved left or right to and the goal is to keep the stick from falling over. A positive reward of +1 is received for every time step that the stick is upright. When it falls past a certain degree then the "episode" is over and a new one can begin. CartPole can be found in [OpenAI Gym's](https://gym.openai.com) list of trainable environments.
This project requires **Python 3.5** with the [Gym](https://gym.openai.com/docs), [numpy, and matplotlib](https://scipy.org/install.html)  libraries installed.
The CartPole-V0 environment is used.

In 'agent.py', there are 3 agent classes defined, each with a different algorithm attached. The basic agent simply moves the cart left if the stick is leaning to the left and moves the cart right if the stick is leaning to the right. The random agent randomly chooses an action (left or right) for the cart to move at every time step. The Q-Learning agent uses a [Q-Learning](https://en.wikipedia.org/wiki/Q-learning) algorithm to choose the best action given the current observation of the cartpole.

When using Q-learning, 'stats.py' plots the agent parameters (alpha & epsilon), rewards per trial, and rolling average rewards per trial. This is useful for visualizing how well the algorithm is performing.

![plot](https://github.com/enerrio/CartPole-Reinforcement-Learning/blob/master/plots.png)

### Usage

In a terminal or command window, navigate to the project directory `CartPole-Reinforcement-Learning/` (that contains this README) and run one of the following commands:

```python3 cartpole.py -a basic```

```python3 cartpole.py -a random```

```python3 cartpole.py -a q-learning```

```python3 cartpole.py --help ```

This will run the `cartpole.py` file with the given agent. Leaving the -a flag empty defaults to a basic agent.
## Extensions to Original Work

### PID Learning (Added by Davin Dillon)
Extended the original Q-learning implementation with a self-tuning PID controller approach:
- `AgentPIDLearning` class with online parameter optimization
- Learns optimal PID gains (Kp, Ki, Kd, Kx) through reinforcement learning
- **Major improvement over Q-learning:** Consistently achieving 3000+ steps after a few thousand training episodes (compared to Q-learning's ~200 step ceiling)

### Human-Initialized Learning Pipeline (October 2025)
Further accelerated PID learning by bootstrapping from human demonstrations:

**What was added:**
- Gamepad demonstration collection (`gamepad_demo_collector.py`) 
- Inverse optimal control parameter fitting (`pid_fitting.py`)
- Human initialization option in `cartpole.py`
- Performance comparison tools (`compare_learning.py`)

**Results (tested at max_steps=5000 to properly evaluate PID performance):**
- **Random-init PID:** ~3400 steps average after 4000 episodes
- **Human-init PID:** 5000 steps (perfect) by episode 400
- Human demos: 15 episodes averaging 45 steps → 77.99% parameter fit accuracy
- **~10x faster convergence** with human bootstrapping

**Two-stage improvement:**
1. Q-learning → PID learning: ~10x improvement (200 → 3000+ steps)
2. Random PID → Human-init PID: ~10x faster convergence to optimal

**Usage:**
```bash
# Collect demonstrations with gamepad
python gamepad_demo_collector.py -n 15

# Extract PID parameters via inverse optimal control
python pid_fitting.py

# Train with human initialization
python cartpole.py -a pid-learning --use-human-init fitted_pid_params.pkl --max-steps 5000

# Compare to random initialization
python cartpole.py -a pid-learning --max-steps 5000