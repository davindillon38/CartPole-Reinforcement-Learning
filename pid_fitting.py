#!/usr/bin/env python
"""
Fit PID parameters from human demonstrations.
Uses optimization to find PID gains that best match human behavior.
"""

import numpy as np
import pickle
import argparse
from scipy.optimize import minimize, differential_evolution
import os
import glob


def load_demonstrations(demo_file):
    """Load demonstrations from pickle file."""
    with open(demo_file, 'rb') as f:
        demos = pickle.load(f)
    
    print(f"Loaded {len(demos)} demonstrations from {demo_file}")
    
    # Print summary
    rewards = [d['total_reward'] for d in demos]
    steps = [d['steps'] for d in demos]
    print(f"  Average reward: {np.mean(rewards):.2f}")
    print(f"  Average steps: {np.mean(steps):.2f}")
    
    return demos


def simulate_pid_action(obs, Kp, Ki, Kd, Kx, cumulative_theta):
    """
    Simulate what action a PID controller would take.
    Returns: action (0 or 1), updated cumulative_theta
    """
    x, x_dot, theta, theta_dot = obs
    
    # Update integral
    cumulative_theta += theta
    
    # PID calculation with position feedback
    pid_value = (Kp * theta + 
                 Ki * cumulative_theta + 
                 Kd * theta_dot +
                 Kx * x)
    
    # Convert to action
    action = 1 if pid_value > 0 else 0
    
    return action, cumulative_theta


def evaluate_pid_params(params, demonstrations, verbose=False):
    """
    Evaluate how well PID parameters match human demonstrations.
    Returns: error (lower is better)
    """
    Kp, Ki, Kd, Kx = params
    
    total_error = 0
    total_actions = 0
    
    for demo in demonstrations:
        observations = demo['observations']
        human_actions = demo['actions']
        
        cumulative_theta = 0
        
        for obs, human_action in zip(observations, human_actions):
            # Simulate PID action
            pid_action, cumulative_theta = simulate_pid_action(
                obs, Kp, Ki, Kd, Kx, cumulative_theta
            )
            
            # Compare to human action
            if pid_action != human_action:
                total_error += 1
            
            total_actions += 1
    
    # Return error rate
    error_rate = total_error / total_actions if total_actions > 0 else 1.0
    
    if verbose:
        accuracy = (1 - error_rate) * 100
        print(f"  Params: Kp={Kp:.4f}, Ki={Ki:.5f}, Kd={Kd:.4f}, Kx={Kx:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
    
    return error_rate


def fit_pid_parameters(demonstrations, method='differential_evolution'):
    """
    Fit PID parameters to match human demonstrations.
    
    Args:
        demonstrations: List of demo episodes
        method: 'differential_evolution' (global) or 'nelder-mead' (local)
    
    Returns:
        Optimal (Kp, Ki, Kd, Kx) parameters
    """
    print("\n" + "="*60)
    print("FITTING PID PARAMETERS FROM HUMAN DEMONSTRATIONS")
    print("="*60)
    print(f"Optimization method: {method}")
    
    # Define bounds for parameters
    bounds = [
        (0.01, 50.0),   # Kp
        (0.0, 2.0),     # Ki
        (0.01, 50.0),   # Kd
        (0.0, 10.0)     # Kx
    ]
    
    if method == 'differential_evolution':
        print("\nRunning global optimization (this may take a minute)...")
        
        result = differential_evolution(
            evaluate_pid_params,
            bounds,
            args=(demonstrations,),
            maxiter=100,
            popsize=15,
            seed=42,
            polish=True,
            workers=1,
            updating='deferred',
            disp=True
        )
        
        optimal_params = result.x
        final_error = result.fun
        
    elif method == 'nelder-mead':
        print("\nRunning local optimization...")
        
        # Start from reasonable initial guess
        x0 = [1.0, 0.01, 1.0, 0.5]
        
        result = minimize(
            evaluate_pid_params,
            x0,
            args=(demonstrations,),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'disp': True}
        )
        
        optimal_params = result.x
        final_error = result.fun
        
        # Clip to bounds
        optimal_params = np.clip(optimal_params, 
                                 [b[0] for b in bounds],
                                 [b[1] for b in bounds])
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    Kp, Ki, Kd, Kx = optimal_params
    accuracy = (1 - final_error) * 100
    
    print(f"Optimal PID parameters:")
    print(f"  Kp = {Kp:.6f}")
    print(f"  Ki = {Ki:.6f}")
    print(f"  Kd = {Kd:.6f}")
    print(f"  Kx = {Kx:.6f}")
    print(f"\nAction matching accuracy: {accuracy:.2f}%")
    print("="*60)
    
    return optimal_params


def save_fitted_params(params, demo_file, output_file='fitted_pid_params.pkl'):
    """Save fitted parameters with metadata."""
    data = {
        'Kp': params[0],
        'Ki': params[1],
        'Kd': params[2],
        'Kx': params[3],
        'demo_file': demo_file,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nFitted parameters saved to: {output_file}")
    return output_file


def find_latest_demo_file():
    """Find the most recent demonstration file."""
    demo_files = glob.glob('demos/human_demos_*.pkl')
    
    if not demo_files:
        return None
    
    # Sort by modification time
    demo_files.sort(key=os.path.getmtime, reverse=True)
    return demo_files[0]


def test_fitted_params(params, demonstrations):
    """
    Test fitted parameters on demonstrations and show detailed results.
    """
    print("\n" + "="*60)
    print("TESTING FITTED PARAMETERS")
    print("="*60)
    
    Kp, Ki, Kd, Kx = params
    
    for i, demo in enumerate(demonstrations[:5]):  # Test on first 5 demos
        observations = demo['observations']
        human_actions = demo['actions']
        
        cumulative_theta = 0
        matches = 0
        
        for obs, human_action in zip(observations, human_actions):
            pid_action, cumulative_theta = simulate_pid_action(
                obs, Kp, Ki, Kd, Kx, cumulative_theta
            )
            
            if pid_action == human_action:
                matches += 1
        
        accuracy = (matches / len(human_actions)) * 100
        print(f"Demo {i+1}: {matches}/{len(human_actions)} actions matched ({accuracy:.1f}%)")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Fit PID parameters from human demonstrations')
    parser.add_argument('--demo-file', type=str, default=None,
                        help='Path to demonstration pickle file (default: latest in demos/)')
    parser.add_argument('--method', type=str, default='differential_evolution',
                        choices=['differential_evolution', 'nelder-mead'],
                        help='Optimization method (default: differential_evolution)')
    parser.add_argument('--output', type=str, default='fitted_pid_params.pkl',
                        help='Output file for fitted parameters')
    parser.add_argument('--test', action='store_true',
                        help='Test fitted parameters on demonstrations')
    args = parser.parse_args()
    
    # Find demo file
    if args.demo_file is None:
        args.demo_file = find_latest_demo_file()
        if args.demo_file is None:
            print("Error: No demonstration files found in demos/")
            print("Please run gamepad_demo_collector.py first")
            return
        print(f"Using latest demo file: {args.demo_file}\n")
    
    # Load demonstrations
    demonstrations = load_demonstrations(args.demo_file)
    
    # Fit parameters
    optimal_params = fit_pid_parameters(demonstrations, method=args.method)
    
    # Test if requested
    if args.test:
        test_fitted_params(optimal_params, demonstrations)
    
    # Save parameters
    save_fitted_params(optimal_params, args.demo_file, args.output)
    
    print("\n✓ PID parameter fitting complete!")
    print(f"\nTo use these parameters in your cartpole.py:")
    print(f"  python cartpole.py -a pid-learning --use-human-init {args.output}")


if __name__ == '__main__':
    main()
