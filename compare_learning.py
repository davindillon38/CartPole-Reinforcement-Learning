#!/usr/bin/env python
"""
Compare learning curves between random initialization and human initialization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os


def load_agent_history(filename):
    """Load agent and extract training history if available."""
    with open(filename, 'rb') as f:
        agent_dict = pickle.load(f)
    
    # If it's a full agent object, return its dict
    if isinstance(agent_dict, dict):
        return agent_dict
    return None


def find_run_files(pattern='agents/pid_agent_run*.pkl'):
    """Find all agent files matching pattern."""
    return sorted(glob.glob(pattern))


def plot_learning_curves(random_init_data, human_init_data, output_file='learning_comparison.png'):
    """
    Plot comparison of learning curves.
    
    Args:
        random_init_data: dict with 'rewards' key containing episode rewards
        human_init_data: dict with 'rewards' key containing episode rewards
        output_file: filename to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PID Learning: Random Init vs Human Init', fontsize=16, fontweight='bold')
    
    # Extract data
    random_rewards = random_init_data['rewards']
    human_rewards = human_init_data['rewards']
    
    window = 100
    
    # Plot 1: Raw rewards
    ax = axes[0, 0]
    ax.plot(random_rewards, alpha=0.3, color='blue', label='Random Init')
    ax.plot(human_rewards, alpha=0.3, color='green', label='Human Init')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Raw Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Moving average
    ax = axes[0, 1]
    
    if len(random_rewards) >= window:
        random_ma = np.convolve(random_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(random_rewards)), random_ma, 
                color='blue', linewidth=2, label=f'Random Init ({window}-ep MA)')
    
    if len(human_rewards) >= window:
        human_ma = np.convolve(human_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(human_rewards)), human_ma, 
                color='green', linewidth=2, label=f'Human Init ({window}-ep MA)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title(f'{window}-Episode Moving Average')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=195, color='red', linestyle='--', alpha=0.5, label='Success threshold')
    
    # Plot 3: Cumulative reward
    ax = axes[1, 0]
    ax.plot(np.cumsum(random_rewards), color='blue', linewidth=2, label='Random Init')
    ax.plot(np.cumsum(human_rewards), color='green', linewidth=2, label='Human Init')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Distribution comparison
    ax = axes[1, 1]
    ax.hist(random_rewards, bins=30, alpha=0.5, color='blue', label='Random Init', density=True)
    ax.hist(human_rewards, bins=30, alpha=0.5, color='green', label='Human Init', density=True)
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Density')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.show()


def print_statistics(random_rewards, human_rewards):
    """Print comparison statistics."""
    print("\n" + "="*70)
    print("LEARNING COMPARISON STATISTICS")
    print("="*70)
    
    print("\n--- Random Initialization ---")
    print(f"Total episodes: {len(random_rewards)}")
    print(f"Mean reward: {np.mean(random_rewards):.2f}")
    print(f"Std reward: {np.std(random_rewards):.2f}")
    print(f"Final 100 episodes mean: {np.mean(random_rewards[-100:]):.2f}")
    print(f"Best episode: {np.max(random_rewards):.1f}")
    
    # Find when it reached 195 average
    window = 100
    if len(random_rewards) >= window:
        ma = np.convolve(random_rewards, np.ones(window)/window, mode='valid')
        solved_idx = np.where(ma >= 195)[0]
        if len(solved_idx) > 0:
            print(f"Reached 195 avg at episode: {solved_idx[0] + window}")
        else:
            print(f"Did not reach 195 average")
    
    print("\n--- Human Initialization ---")
    print(f"Total episodes: {len(human_rewards)}")
    print(f"Mean reward: {np.mean(human_rewards):.2f}")
    print(f"Std reward: {np.std(human_rewards):.2f}")
    print(f"Final 100 episodes mean: {np.mean(human_rewards[-100:]):.2f}")
    print(f"Best episode: {np.max(human_rewards):.1f}")
    
    if len(human_rewards) >= window:
        ma = np.convolve(human_rewards, np.ones(window)/window, mode='valid')
        solved_idx = np.where(ma >= 195)[0]
        if len(solved_idx) > 0:
            print(f"Reached 195 avg at episode: {solved_idx[0] + window}")
        else:
            print(f"Did not reach 195 average")
    
    # Comparison
    print("\n--- Comparison ---")
    improvement = np.mean(human_rewards[:100]) - np.mean(random_rewards[:100])
    print(f"First 100 episodes - Human advantage: {improvement:+.2f} reward")
    
    if len(random_rewards) >= 100 and len(human_rewards) >= 100:
        final_improvement = np.mean(human_rewards[-100:]) - np.mean(random_rewards[-100:])
        print(f"Final 100 episodes - Human advantage: {final_improvement:+.2f} reward")
    
    print("="*70)


def create_mock_data_for_demo():
    """Create mock training data for demonstration if no real data available."""
    print("\n[Demo Mode] Creating synthetic data for visualization...")
    
    # Random init: starts low, gradually improves
    random_rewards = []
    for i in range(1000):
        base = min(50 + i * 0.15, 200)
        noise = np.random.randn() * 30
        random_rewards.append(max(10, base + noise))
    
    # Human init: starts higher, improves faster
    human_rewards = []
    for i in range(1000):
        base = min(120 + i * 0.08, 200)
        noise = np.random.randn() * 25
        human_rewards.append(max(50, base + noise))
    
    return {
        'rewards': random_rewards,
        'source': 'synthetic_random'
    }, {
        'rewards': human_rewards,
        'source': 'synthetic_human'
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare random vs human-initialized learning')
    parser.add_argument('--random-file', type=str, default=None,
                        help='Pickle file with random init training data')
    parser.add_argument('--human-file', type=str, default=None,
                        help='Pickle file with human init training data')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode with synthetic data')
    parser.add_argument('--output', type=str, default='learning_comparison.png',
                        help='Output filename for plot')
    args = parser.parse_args()
    
    if args.demo:
        # Create synthetic data for demonstration
        random_data, human_data = create_mock_data_for_demo()
    else:
        # Try to find files automatically if not specified
        if args.random_file is None:
            random_files = glob.glob('agents/pid_agent_run*_final.pkl')
            random_files = [f for f in random_files if 'human_init' not in f]
            if random_files:
                args.random_file = random_files[-1]
                print(f"Using random init file: {args.random_file}")
        
        if args.human_file is None:
            human_files = glob.glob('agents/pid_agent_run*_human_init_final.pkl')
            if human_files:
                args.human_file = human_files[-1]
                print(f"Using human init file: {args.human_file}")
        
        # Check if files exist
        if args.random_file is None or args.human_file is None:
            print("\nError: Could not find training data files.")
            print("\nOptions:")
            print("  1. Specify files manually: --random-file <file> --human-file <file>")
            print("  2. Run demo mode: --demo")
            print("\nTo generate real data:")
            print("  1. Collect demos: python gamepad_demo_collector.py")
            print("  2. Fit parameters: python pid_fitting.py")
            print("  3. Train random: python cartpole.py -a pid-learning")
            print("  4. Train human-init: python cartpole.py -a pid-learning --use-human-init fitted_pid_params.pkl")
            return
        
        # Note: The actual implementation would need to save training histories
        # This is a template - you'd need to modify your cartpole.py to save
        # the 'totals' array along with the agent
        print("\nNote: This script expects training data to be saved.")
        print("You may need to modify your cartpole.py to save training histories.")
        print("\nRun with --demo flag to see example visualization.")
        
        return
    
    # Print statistics and plot
    print_statistics(random_data['rewards'], human_data['rewards'])
    plot_learning_curves(random_data, human_data, args.output)
    
    print("\n✓ Comparison complete!")


if __name__ == '__main__':
    main()