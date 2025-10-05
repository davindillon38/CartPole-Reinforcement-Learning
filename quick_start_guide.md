# CartPole Quick Start Guide

Complete guide for using all agents and workflows in this repository.

## Table of Contents
- [Quick Start Examples](#quick-start-examples)
- [Command-Line Arguments](#command-line-arguments-reference)
- [File Structure](#project-file-structure)
- [Workflow Details](#workflow-details)
- [Troubleshooting](#troubleshooting)

---

## Quick Start Examples

### Original Q-Learning (from base repo)
```bash
python cartpole.py -a q-learning --episodes 2000
```

### PID Learning with Random Initialization
```bash
python cartpole.py -a pid-learning --max-steps 5000 --episodes 10000
```

### Human-Initialized PID Learning (Complete Workflow)

**Step 1: Collect Human Demonstrations**
```bash
python gamepad_demo_collector.py -n 15 --fps 10
```
- Requires Logitech F310 gamepad (or compatible controller)
- Controls: D-pad left/right to move cart, A button=skip episode, X button=save & exit
- Adjust `--fps` to control game speed (lower = easier to play)
- Demonstrations saved to `demos/` folder

**Step 2: Fit PID Parameters from Demonstrations**
```bash
python pid_fitting.py
```
- Automatically uses latest demo file from `demos/`
- Uses differential evolution to optimize PID parameters
- Outputs: `fitted_pid_params.pkl` with extracted parameters
- Displays action matching accuracy (target: >75%)

**Step 3: Train with Human Initialization**
```bash
python cartpole.py -a pid-learning --use-human-init fitted_pid_params.pkl --max-steps 5000 -n "Human Init Run"
```

**Step 4: Compare with Random Initialization**
```bash
# Run random init for comparison
python cartpole.py -a pid-learning --max-steps 5000 -n "Random Init Run"

# View comparison (demo mode)
python compare_learning.py --demo
```

---

## Command-Line Arguments Reference

### cartpole.py
```bash
python cartpole.py [options]
```

**Required:**
- `-a, --agent {basic,random,q-learning,pid-learning}` - Agent type

**Optional:**
- `-n, --run-name TEXT` - W&B run name for tracking experiments
- `--max-steps INT` - Maximum steps per episode (default: 300)
- `--episodes INT` - Number of training episodes (default: 17000 for q-learning, 10000 for PID)
- `--run-number INT` - Specific run number for file naming (auto-increments if not specified)
- `--use-human-init PATH` - Path to fitted PID parameters file (enables human initialization)

**Examples:**
```bash
# Q-learning with custom episode count
python cartpole.py -a q-learning --episodes 5000 -n "Q-Learning Test"

# PID with human initialization
python cartpole.py -a pid-learning --use-human-init fitted_pid_params.pkl --max-steps 5000

# PID with random initialization and specific run number
python cartpole.py -a pid-learning --run-number 10 --max-steps 300
```

### gamepad_demo_collector.py
```bash
python gamepad_demo_collector.py [options]
```

**Optional:**
- `-n, --num-episodes INT` - Number of demonstration episodes to collect (default: 10)
- `--fps INT` - Frames per second for gameplay (default: 10, range: 5-20, lower is easier)

**Examples:**
```bash
# Collect 20 episodes at slow speed
python gamepad_demo_collector.py -n 20 --fps 8

# Collect 10 episodes at default speed
python gamepad_demo_collector.py
```

### pid_fitting.py
```bash
python pid_fitting.py [options]
```

**Optional:**
- `--demo-file PATH` - Path to specific demonstration file (default: latest in demos/)
- `--method {differential_evolution,nelder-mead}` - Optimization method (default: differential_evolution)
- `--output PATH` - Output filename for fitted parameters (default: fitted_pid_params.pkl)
- `--test` - Test fitted parameters on demonstrations after fitting

**Examples:**
```bash
# Fit using specific demo file
python pid_fitting.py --demo-file demos/human_demos_20251003_083851.pkl

# Fit with testing enabled
python pid_fitting.py --test

# Use local optimization method
python pid_fitting.py --method nelder-mead
```

### compare_learning.py
```bash
python compare_learning.py [options]
```

**Optional:**
- `--random-file PATH` - Path to random init training data
- `--human-file PATH` - Path to human init training data  
- `--demo` - Run in demo mode with synthetic data
- `--output PATH` - Output filename for comparison plot (default: learning_comparison.png)

**Examples:**
```bash
# View demo comparison
python compare_learning.py --demo

# Compare specific runs
python compare_learning.py --random-file agents/pid_agent_run001_final.pkl --human-file agents/pid_agent_run002_human_init_final.pkl
```

---

## Project File Structure

```
CartPole-Reinforcement-Learning/
├── cartpole.py                    # Main training script (modified with human-init support)
├── agent.py                       # Agent class definitions (Basic, Random, Q-Learning, PID)
├── stats.py                       # Statistics and plotting utilities
│
├── gamepad_demo_collector.py      # NEW: Collect human demonstrations via gamepad
├── pid_fitting.py                 # NEW: Extract PID parameters using inverse optimal control
├── compare_learning.py            # NEW: Compare learning curves (random vs human-init)
├── cartpole_patch.py              # NEW: Integration code reference
├── cartpole_human_init.py         # NEW: Alternative implementation reference
│
├── demos/                         # NEW: Stored human demonstrations
│   ├── human_demos_TIMESTAMP.pkl
│   └── human_demos_TIMESTAMP_metadata.json
│
├── agents/                        # NEW: Saved trained agents
│   ├── pid_agent_runXXX_final.pkl
│   ├── pid_agent_runXXX_best.pkl
│   └── pid_agent_runXXX_human_init_final.pkl
│
├── fitted_pid_params.pkl          # NEW: Most recent fitted PID parameters
│
└── wandb/                         # W&B training logs (generated during runs)
```

---

## Workflow Details

### Human Demonstration Collection

The gamepad collector creates a custom pygame window showing:
- Cart and pole visualization
- Current action indicator (LEFT/RIGHT)
- Real-time statistics (step count, reward, angle, position)
- Control hints

**Tips for good demonstrations:**
- Focus on keeping the pole upright
- Try to keep the cart centered
- Make small, frequent corrections rather than large movements
- 10-20 episodes with 30+ steps each provides good data
- Consistency matters more than perfection

### PID Parameter Fitting

Uses inverse optimal control to extract PID gains from your demonstrations:

1. **Forward Model**: Simulates a PID controller with candidate parameters
2. **Objective Function**: Counts mismatches between PID and human actions
3. **Optimization**: Differential evolution searches parameter space to minimize mismatches

**Output interpretation:**
- Action matching >75%: Good fit, parameters capture your strategy
- Action matching 60-75%: Acceptable, may work but less optimal
- Action matching <60%: Poor fit, consider collecting more demos

### Training Comparison

**Random Initialization:**
- Starts with parameters sampled from broad ranges
- Explores parameter space through trial and error
- Can take 3000-5000 episodes to reach good performance
- May get stuck in local optima

**Human Initialization:**
- Starts with parameters fitted from your demonstrations
- Begins in a proven stable region of parameter space
- Reaches good performance in 400-500 episodes
- More consistent results across runs

---

## Expected Workflow Output

**After collecting demos:**
```
demos/human_demos_20251003_083851.pkl
demos/human_demos_20251003_083851_metadata.json

Console output:
  Episodes collected: 15
  Average reward: 45.73
  Total steps: 686
```

**After fitting parameters:**
```
fitted_pid_params.pkl

Console output:
  Optimal PID parameters:
    Kp = 40.779584
    Ki = 0.011456
    Kd = 0.070005
    Kx = 5.089605
  Action matching accuracy: 77.99%
```

**After training:**
```
agents/pid_agent_run001_final.pkl      # Final learned parameters
agents/pid_agent_run001_best.pkl       # Best parameters during training

W&B Dashboard:
  - Real-time training curves
  - Parameter evolution over time
  - Performance metrics
```

---

## Troubleshooting

### Gamepad Issues

**Gamepad not detected:**
```bash
# Test gamepad detection
python -c "import pygame; pygame.init(); pygame.joystick.init(); print(f'Gamepads: {pygame.joystick.get_count()}')"
```
- Ensure Logitech F310 switch is set to "D" (DirectInput mode)
- Check USB connection
- Try different USB port

**D-pad not working:**
- Confirm switch is in "D" mode (not "X")
- The rendering window may steal events - try `--no-render` flag if added

**Game too fast/slow:**
- Adjust `--fps` parameter (recommended range: 5-20)
- Lower values = more time to react
- Try `--fps 5` for very slow, easy control

### Parameter Fitting Issues

**Poor fitting accuracy (<70%):**
- Collect more demonstrations (15-20 recommended)
- Try to be more consistent in control strategy
- Ensure episodes are long enough (aim for 30+ steps)
- Check that demos aren't all very short failures

**Optimization taking too long:**
- Use `--method nelder-mead` for faster local optimization
- Reduce demo file size by using specific file with fewer episodes

### Training Issues

**Agent not improving:**
- Check that `--episodes` is high enough (10000+ for PID)
- Verify `--max-steps` allows enough time (5000 for testing PID ceiling)
- Review W&B logs for parameter evolution
- Try different random seed or run number

**Human-init worse than random:**
- Demonstrations may be too poor quality
- Try collecting new demos with better performance
- Check fitting accuracy - should be >75%

### Git Issues

**Git not recognized in PowerShell:**
```powershell
# Add git to PATH permanently
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\Git\cmd", [EnvironmentVariableTarget]::User)
# Then restart PowerShell
```

**Diverged branches:**
```bash
git pull origin master --rebase
```

**Wandb log conflicts:**
```bash
# Delete temporary logs causing issues
Remove-Item -Recurse -Force wandb/
git pull origin master
```

---

## Performance Benchmarks

Based on actual results from this implementation:

| Method | Initial Performance | Episode 400 | Episode 4000 | Episodes to "Solve" |
|--------|-------------------|-------------|--------------|---------------------|
| Q-Learning | ~20 steps | ~180 steps | ~200 steps | Does not solve |
| Random PID | ~30 steps | ~200 steps | ~3400 steps | ~3000 |
| Human PID | ~4000 steps | ~5000 steps | N/A | ~400 |

*"Solve" defined as 100-episode average ≥195 steps for max_steps=300, or perfect performance for max_steps=5000*

---

## Next Steps

After completing the CartPole experiments:

1. **Analyze results in W&B**: Compare learning curves, parameter evolution
2. **Experiment with variations**: Different demo quantities, fitting methods, learning rates
3. **Apply to TurtleBot3**: Use similar pipeline with behavior cloning instead of PID
4. **Document findings**: Save plots, note parameter values, record observations

For questions or issues, refer to the main README.md or open an issue on GitHub.
