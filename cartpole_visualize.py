#!/usr/bin/env python
"""
Visualize a trained PID agent playing CartPole
Usage: python visualize_agent.py [--agent-file pid_learning_agent_best.pkl] [--episodes 5] [--speed 1.0]
"""

import gym
from gym.wrappers import TimeLimit
import numpy as np
import pickle
import argparse
import time
import glob
import os


def list_available_agents():
    """List all available agent files organized by run number"""
    agent_files = glob.glob("agents/pid_agent_run*_*.pkl")
    
    if not agent_files:
        print("No saved agents found in agents/ folder.")
        return {}
    
    # Organize by run number
    runs = {}
    for filename in agent_files:
        try:
            basename = os.path.basename(filename)
            parts = basename.split('_')
            if len(parts) >= 4 and parts[2].startswith('run'):
                run_num = int(parts[2][3:])
                agent_type = parts[3].replace('.pkl', '')  # 'best' or 'final'
                
                if run_num not in runs:
                    runs[run_num] = {}
                runs[run_num][agent_type] = filename
        except (ValueError, IndexError):
            continue
    
    return runs


def print_available_agents():
    """Print a nice list of available agents"""
    runs = list_available_agents()
    
    if not runs:
        print("\nNo saved agents found in agents/ folder.")
        print("Train an agent first with: python cartpole.py --agent pid-learning\n")
        return
    
    print("\n" + "="*60)
    print("AVAILABLE AGENTS (in agents/ folder)")
    print("="*60)
    for run_num in sorted(runs.keys()):
        print(f"\nRun {run_num:03d}:")
        if 'best' in runs[run_num]:
            print(f"  --run {run_num}  (loads {runs[run_num]['best']})")
        if 'final' in runs[run_num]:
            print(f"  --agent-file {runs[run_num]['final']}")
    print("="*60 + "\n")


class AgentPIDLearning:
    """
    PID Learning Agent for CartPole.
    This is a minimal version just for running inference (no learning).
    """
    def __init__(self, Kp=1.0, Ki=0.0, Kd=1.0, Kx=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kx = Kx  # Cart position gain
        self.cumulative_theta = 0
        
    def reset_episode(self):
        """Reset integrator for new episode"""
        self.cumulative_theta = 0

    def act(self, obs):
        """PID controller action with cart position feedback"""
        x, x_dot, theta, theta_dot = obs
        
        # PID control with cart position term
        self.cumulative_theta += theta
        pid_value = (self.Kp * theta + 
                    self.Ki * self.cumulative_theta + 
                    self.Kd * theta_dot +
                    self.Kx * x)  # Position feedback
        
        # Convert to action
        action = 1 if pid_value > 0 else 0
        return action

    def load(self, filename):
        """Load agent parameters from file"""
        # Check agents/ folder if not already in path
        if not os.path.exists(filename):
            if not filename.startswith("agents/"):
                test_path = os.path.join("agents", filename)
                if os.path.exists(test_path):
                    filename = test_path
        
        with open(filename, "rb") as f:
            saved_dict = pickle.load(f)
            self.Kp = saved_dict['Kp']
            self.Ki = saved_dict['Ki']
            self.Kd = saved_dict['Kd']
            self.Kx = saved_dict.get('Kx', 0.0)  # Default to 0 for old agents
            if 'best_reward' in saved_dict:
                self.best_reward = saved_dict['best_reward']
        print(f"\nLoaded PID agent from {filename}")
        print(f"Parameters: Kp={self.Kp:.4f}, Ki={self.Ki:.5f}, Kd={self.Kd:.4f}, Kx={self.Kx:.4f}")
        if hasattr(self, 'best_reward'):
            print(f"Best training reward: {self.best_reward:.1f}")
        print()


def run_visual_episode(env, agent, episode_num, render_speed=1.0, max_steps=500):
    """
    Run a single episode with visual rendering.
    
    Args:
        env: Gym environment
        agent: PID agent
        episode_num: Episode number for display
        render_speed: Speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
        max_steps: Maximum steps for this episode
    
    Returns:
        total_reward: Total steps survived
    """
    obs, _ = env.reset()
    agent.reset_episode()
    
    total_reward = 0
    step = 0
    
    print(f"\n{'='*60}")
    print(f"Episode {episode_num} (max steps: {max_steps})")
    print(f"{'='*60}")
    
    # Calculate frame delay based on render speed
    # Normal CartPole runs at ~50 FPS, so 0.02 seconds per frame
    frame_delay = 0.02 / render_speed
    
    while True:
        # Render the environment
        env.render()
        
        # Get action from agent
        action = agent.act(obs)
        
        # Display current state info
        x, x_dot, theta, theta_dot = obs
        pid_value = agent.Kp * theta + agent.Ki * agent.cumulative_theta + agent.Kd * theta_dot + agent.Kx * x
        
        if step % 10 == 0:  # Print every 10 steps to avoid spam
            print(f"Step {step:3d} | "
                  f"Cart pos: {x:6.3f} | "
                  f"Pole angle: {theta:7.4f} | "
                  f"Angular vel: {theta_dot:7.4f} | "
                  f"PID: {pid_value:7.4f} | "
                  f"Action: {'RIGHT' if action == 1 else 'LEFT '}")
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
        
        # Control rendering speed
        time.sleep(frame_delay)
        
        if done:
            print(f"\n{'*'*60}")
            print(f"Episode ended at step {step}")
            if step >= max_steps:
                print("SUCCESS! Maximum steps reached!")
            elif abs(x) > 2.4:
                print("FAILED: Cart went off screen")
            else:
                print("FAILED: Pole fell over")
            print(f"Total reward: {total_reward:.0f}")
            print(f"{'*'*60}\n")
            break
    
    return total_reward


def run_batch_episodes(env, agent, num_episodes=100, max_steps=500):
    """
    Run multiple episodes without rendering to get statistics.
    
    Args:
        env: Gym environment (without render mode)
        agent: PID agent
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    
    Returns:
        rewards: List of rewards for each episode
    """
    rewards = []
    
    print(f"\nRunning {num_episodes} test episodes (no rendering, max {max_steps} steps)...")
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        agent.reset_episode()
        
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
        
        if (ep + 1) % 10 == 0:
            avg_so_far = np.mean(rewards)
            print(f"  Completed {ep + 1}/{num_episodes} episodes. Average so far: {avg_so_far:.2f}")
    
    return rewards


def main():
    parser = argparse.ArgumentParser(description='Visualize trained PID agent on CartPole')
    parser.add_argument('--agent-file', type=str, default=None,
                        help='Path to saved agent pickle file (e.g., pid_agent_run001_best.pkl)')
    parser.add_argument('--run', type=int, default=None,
                        help='Run number to load (loads best agent from that run)')
    parser.add_argument('--list', action='store_true',
                        help='List all available saved agents and exit')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to visualize')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed (1.0=normal, 0.5=half speed, 2.0=double speed)')
    parser.add_argument('--test', action='store_true',
                        help='Run 100 test episodes without rendering to get statistics')
    parser.add_argument('--manual-params', action='store_true',
                        help='Use manual PID parameters instead of loading from file')
    parser.add_argument('--kp', type=float, default=1.0,
                        help='Manual Kp value (only with --manual-params)')
    parser.add_argument('--ki', type=float, default=0.0,
                        help='Manual Ki value (only with --manual-params)')
    parser.add_argument('--kd', type=float, default=1.0,
                        help='Manual Kd value (only with --manual-params)')
    parser.add_argument('--kx', type=float, default=0.5,
                        help='Manual Kx value for cart position (only with --manual-params)')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode (default: 500)')
    
    args = parser.parse_args()
    
    # Handle --list flag
    if args.list:
        print_available_agents()
        return
    
    # Create agent
    agent = AgentPIDLearning()
    
    if args.manual_params:
        agent.Kp = args.kp
        agent.Ki = args.ki
        agent.Kd = args.kd
        agent.Kx = args.kx
        print(f"\nUsing manual PID parameters:")
        print(f"Kp={agent.Kp:.4f}, Ki={agent.Ki:.5f}, Kd={agent.Kd:.4f}, Kx={agent.Kx:.4f}\n")
    else:
        # Determine which file to load
        agent_file = None
        
        if args.agent_file:
            # Explicit file specified
            agent_file = args.agent_file
            # Add agents/ prefix if not present and file doesn't exist
            if not os.path.exists(agent_file) and not agent_file.startswith("agents/"):
                test_path = os.path.join("agents", agent_file)
                if os.path.exists(test_path):
                    agent_file = test_path
        elif args.run:
            # Run number specified - load best from that run
            agent_file = f"agents/pid_agent_run{args.run:03d}_best.pkl"
            if not os.path.exists(agent_file):
                print(f"Error: Agent file '{agent_file}' not found for run {args.run}")
                print_available_agents()
                return
        else:
            # No specification - try to find most recent run
            runs = list_available_agents()
            if runs:
                latest_run = max(runs.keys())
                if 'best' in runs[latest_run]:
                    agent_file = runs[latest_run]['best']
                    print(f"No agent specified, using latest: {agent_file}")
                else:
                    agent_file = runs[latest_run]['final']
                    print(f"No agent specified, using latest: {agent_file}")
            else:
                print("Error: No saved agents found")
                print("Train an agent first with: python cartpole.py --agent pid-learning")
                print("Or use --manual-params to specify PID values manually")
                return
        
        # Load the agent
        try:
            agent.load(agent_file)
        except FileNotFoundError:
            print(f"Error: Could not find agent file '{agent_file}'")
            print_available_agents()
            return
        except Exception as e:
            print(f"Error loading agent: {e}")
            return
    
    if args.test:
        # Run batch testing without rendering
        # Create base environment and wrap with custom time limit
        base_env = gym.make('CartPole-v1').unwrapped
        env = TimeLimit(base_env, max_episode_steps=args.max_steps)
        
        rewards = run_batch_episodes(env, agent, num_episodes=100, max_steps=args.max_steps)
        
        success_threshold = int(0.65 * args.max_steps)
        
        print(f"\n{'='*60}")
        print("TEST RESULTS")
        print(f"{'='*60}")
        print(f"Episodes run: {len(rewards)}")
        print(f"Max steps: {args.max_steps}")
        print(f"Average reward: {np.mean(rewards):.2f}")
        print(f"Std deviation: {np.std(rewards):.2f}")
        print(f"Min reward: {np.min(rewards):.0f}")
        print(f"Max reward: {np.max(rewards):.0f}")
        print(f"Median reward: {np.median(rewards):.2f}")
        print(f"Success rate (≥{success_threshold}): {100 * np.sum(np.array(rewards) >= success_threshold) / len(rewards):.1f}%")
        print(f"Perfect rate (={args.max_steps}): {100 * np.sum(np.array(rewards) == args.max_steps) / len(rewards):.1f}%")
        print(f"{'='*60}\n")
        
        env.close()
    else:
        # Run visual episodes
        print("\n" + "="*60)
        print("CARTPOLE PID AGENT VISUALIZATION")
        print("="*60)
        print(f"Running {args.episodes} episode(s) with rendering")
        print(f"Max steps per episode: {args.max_steps}")
        print(f"Playback speed: {args.speed}x")
        print("Close the window or press Ctrl+C to exit early")
        print("="*60)
        
        # Create environment with rendering and custom time limit
        base_env = gym.make('CartPole-v1', render_mode='human').unwrapped
        env = TimeLimit(base_env, max_episode_steps=args.max_steps)
        
        rewards = []
        
        try:
            for ep in range(args.episodes):
                reward = run_visual_episode(env, agent, ep + 1, render_speed=args.speed, max_steps=args.max_steps)
                rewards.append(reward)
                
                # Small pause between episodes
                if ep < args.episodes - 1:
                    print("\nStarting next episode in 2 seconds...")
                    time.sleep(2)
            
            # Print summary
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Episodes completed: {len(rewards)}")
            print(f"Average reward: {np.mean(rewards):.2f}")
            print(f"Rewards: {[int(r) for r in rewards]}")
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nVisualization interrupted by user")
        finally:
            env.close()
            print("Environment closed")


if __name__ == '__main__':
    main()
