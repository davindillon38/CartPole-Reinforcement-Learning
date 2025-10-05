#!/usr/bin/env python
"""
Modified cartpole.py with human-initialized PID learning.
Add this code to your existing cartpole.py or use as reference.
"""

import pickle
import os

# Add these functions to your existing cartpole.py

def load_human_pid_params(params_file):
    """Load human-fitted PID parameters from file."""
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Parameter file not found: {params_file}")
    
    with open(params_file, 'rb') as f:
        data = pickle.load(f)
    
    Kp = data['Kp']
    Ki = data['Ki']
    Kd = data['Kd']
    Kx = data['Kx']
    
    print("\n" + "="*60)
    print("LOADED HUMAN-FITTED PID PARAMETERS")
    print("="*60)
    print(f"Source: {data.get('demo_file', 'unknown')}")
    print(f"Kp = {Kp:.6f}")
    print(f"Ki = {Ki:.6f}")
    print(f"Kd = {Kd:.6f}")
    print(f"Kx = {Kx:.6f}")
    print("="*60 + "\n")
    
    return Kp, Ki, Kd, Kx


def pid_learning_with_human_init(env, params_file, episodes=10000, max_steps=300, run_number=None):
    """
    Train PID controller starting from human-fitted parameters.
    
    Args:
        env: Gym environment
        params_file: Path to fitted PID parameters
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        run_number: Run number for saving files
    """
    # Load human parameters
    Kp, Ki, Kd, Kx = load_human_pid_params(params_file)
    
    # Get run number if not provided
    if run_number is None:
        run_number = get_next_run_number()
    
    # Set the max episode steps
    env._max_episode_steps = max_steps
    
    # Initialize agent with human parameters
    agent = AgentPIDLearning(Kp=Kp, Ki=Ki, Kd=Kd, Kx=Kx, lr=0.05, max_steps=max_steps)
    
    totals = []
    window_size = 100
    solved = False
    
    # Calculate success threshold (65% of max_steps)
    success_threshold = int(0.65 * max_steps)

    print("Starting PID Learning with Human Initialization...")
    print(f"Run number: {run_number}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Success threshold: {success_threshold} steps average")
    print(f"Initial params: Kp={agent.Kp:.4f}, Ki={agent.Ki:.5f}, Kd={agent.Kd:.4f}, Kx={agent.Kx:.4f}")
    print("Note: Starting with HUMAN-FITTED parameters\n")
    
    for episode in range(episodes):
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
        
        # Learn from this episode
        agent.learn_from_episode(episode_reward)
        totals.append(episode_reward)
        
        # Logging
        wandb.log({
            "pid_reward": episode_reward,
            "Kp": agent.Kp,
            "Ki": agent.Ki,
            "Kd": agent.Kd,
            "Kx": agent.Kx,
            "learning_rate": agent.lr,
            "best_reward": agent.best_reward,
            "human_initialized": True
        })

        # Progress reporting
        if episode % 100 == 0:
            recent_avg = np.mean(totals[-window_size:]) if len(totals) >= window_size else np.mean(totals)
            print(f"Episode {episode}")
            print(f"  Current reward: {episode_reward:.1f}")
            print(f"  Recent avg ({min(len(totals), window_size)} eps): {recent_avg:.1f}")
            print(f"  Best reward: {agent.best_reward:.1f}")
            print(f"  Params: Kp={agent.Kp:.4f}, Ki={agent.Ki:.5f}, Kd={agent.Kd:.4f}, Kx={agent.Kx:.4f}")
            print(f"  Learning rate: {agent.lr:.6f}")
        
        # Check if solved
        if len(totals) >= window_size and not solved:
            recent_avg = np.mean(totals[-window_size:])
            if recent_avg >= success_threshold:
                print(f"\n{'='*50}")
                print(f"SOLVED at episode {episode}!")
                print(f"Average reward over last {window_size} episodes: {recent_avg:.2f}")
                print(f"Success threshold was: {success_threshold}")
                print(f"Final params: Kp={agent.Kp:.4f}, Ki={agent.Ki:.5f}, Kd={agent.Kd:.4f}, Kx={agent.Kx:.4f}")
                print(f"{'='*50}\n")
                solved = True

    # Create filenames with run number and human-init marker
    final_filename = f"agents/pid_agent_run{run_number:03d}_human_init_final.pkl"
    best_filename = f"agents/pid_agent_run{run_number:03d}_human_init_best.pkl"
    
    # Ensure agents directory exists
    os.makedirs("agents", exist_ok=True)
    
    # Save both the final agent and the best-performing agent
    agent.save(final_filename)
    
    # Also save agent with best parameters
    agent_best = AgentPIDLearning(Kp=agent.best_params[0], 
                                   Ki=agent.best_params[1], 
                                   Kd=agent.best_params[2],
                                   Kx=agent.best_params[3],
                                   max_steps=max_steps)
    agent_best.best_reward = agent.best_reward
    agent_best.save(best_filename)
    
    print("\nPID learning (human-initialized) completed.")
    print(f"Run number: {run_number}")
    print(f"Final average (last 100): {np.mean(totals[-100:]):.2f}")
    print(f"Best single episode: {max(totals):.1f}")
    print(f"Final params: Kp={agent.Kp:.4f}, Ki={agent.Ki:.5f}, Kd={agent.Kd:.4f}, Kx={agent.Kx:.4f}")
    print(f"Best params: Kp={agent.best_params[0]:.4f}, Ki={agent.best_params[1]:.5f}, Kd={agent.best_params[2]:.4f}, Kx={agent.best_params[3]:.4f}")
    print(f"\nSaved agents:")
    print(f"  - {final_filename} (final parameters)")
    print(f"  - {best_filename} (best performing parameters)")
    
    return totals


# ADD TO YOUR ARGUMENT PARSER IN main():
"""
parser.add_argument('--use-human-init', type=str, default=None,
                    help='Path to human-fitted PID parameters file')
"""

# ADD TO YOUR main() FUNCTION AFTER pid-learning CASE:
"""
elif args.agent == 'pid-learning':
    if args.use_human_init:
        # Human-initialized PID learning
        totals = pid_learning_with_human_init(
            env, 
            args.use_human_init,
            max_steps=args.max_steps, 
            run_number=args.run_number
        )
    else:
        # Regular PID learning (random initialization)
        totals = pid_learning(env, max_steps=args.max_steps, run_number=args.run_number)
    
    print(f"\nFinal Statistics:")
    print(f"  Average reward (all episodes): {np.mean(totals):.2f}")
    print(f"  Average reward (last 100): {np.mean(totals[-100:]):.2f}")
    print(f"  Std dev (last 100): {np.std(totals[-100:]):.2f}")
    wandb.finish()
"""
