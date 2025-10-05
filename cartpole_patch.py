#!/usr/bin/env python
"""
Patch to add human-initialization support to existing cartpole.py

INSTRUCTIONS:
Add these sections to your existing cartpole.py file
"""

# =============================================================================
# SECTION 1: Add to imports (near the top of the file)
# =============================================================================
# (Already have: import pickle, os, glob)


# =============================================================================
# SECTION 2: Add this function anywhere before main()
# =============================================================================

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


# =============================================================================
# SECTION 3: Modify pid_learning() function signature
# Add optional init_params parameter
# =============================================================================

def pid_learning(env, episodes=10000, max_steps=300, run_number=None, init_params=None):
    """
    Train PID controller through trial and error.
    
    Args:
        env: Gym environment
        episodes: Number of training episodes
        max_steps: Maximum steps per episode (default 300)
        run_number: Run number for saving files (auto-generated if None)
        init_params: Tuple of (Kp, Ki, Kd, Kx) for initialization (optional)
    """
    # Get run number if not provided
    if run_number is None:
        run_number = get_next_run_number()
    
    # Set the max episode steps
    env._max_episode_steps = max_steps
    
    # Initialize agent with provided parameters or random
    if init_params is not None:
        Kp, Ki, Kd, Kx = init_params
        agent = AgentPIDLearning(Kp=Kp, Ki=Ki, Kd=Kd, Kx=Kx, lr=0.05, max_steps=max_steps)
        init_method = "human-initialized"
        print(f"Starting with HUMAN-FITTED parameters")
    else:
        agent = AgentPIDLearning(lr=0.05, max_steps=max_steps)
        init_method = "random"
        print(f"Starting with RANDOM parameters")
    
    totals = []
    window_size = 100
    solved = False
    
    # Calculate success threshold (65% of max_steps)
    success_threshold = int(0.65 * max_steps)

    print("Starting PID Learning...")
    print(f"Run number: {run_number}")
    print(f"Initialization: {init_method}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Success threshold: {success_threshold} steps average")
    print(f"Initial params: Kp={agent.Kp:.4f}, Ki={agent.Ki:.5f}, Kd={agent.Kd:.4f}, Kx={agent.Kx:.4f}")
    print()
    
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
            "init_method": init_method
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

    # Create filenames with run number and init method
    suffix = "_human_init" if init_params is not None else ""
    final_filename = f"agents/pid_agent_run{run_number:03d}{suffix}_final.pkl"
    best_filename = f"agents/pid_agent_run{run_number:03d}{suffix}_best.pkl"
    
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
    
    print(f"\nPID learning ({init_method}) completed.")
    print(f"Run number: {run_number}")
    print(f"Final average (last 100): {np.mean(totals[-100:]):.2f}")
    print(f"Best single episode: {max(totals):.1f}")
    print(f"Final params: Kp={agent.Kp:.4f}, Ki={agent.Ki:.5f}, Kd={agent.Kd:.4f}, Kx={agent.Kx:.4f}")
    print(f"Best params: Kp={agent.best_params[0]:.4f}, Ki={agent.best_params[1]:.5f}, Kd={agent.best_params[2]:.4f}, Kx={agent.best_params[3]:.4f}")
    print(f"\nSaved agents:")
    print(f"  - {final_filename} (final parameters)")
    print(f"  - {best_filename} (best performing parameters)")
    
    return totals


# =============================================================================
# SECTION 4: Add to argument parser in main()
# =============================================================================
"""
Add this line to your argument parser:

parser.add_argument('--use-human-init', type=str, default=None,
                    help='Path to human-fitted PID parameters file')
"""


# =============================================================================
# SECTION 5: Modify the pid-learning case in main()
# Replace your existing elif args.agent == 'pid-learning': block with this:
# =============================================================================
"""
elif args.agent == 'pid-learning':
    # Check if using human initialization
    init_params = None
    if args.use_human_init:
        try:
            init_params = load_human_pid_params(args.use_human_init)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Falling back to random initialization")
            init_params = None
    
    # Run PID learning with or without human init
    totals = pid_learning(
        env, 
        max_steps=args.max_steps, 
        run_number=args.run_number,
        init_params=init_params
    )
    
    print(f"\nFinal Statistics:")
    print(f"  Average reward (all episodes): {np.mean(totals):.2f}")
    print(f"  Average reward (last 100): {np.mean(totals[-100:]):.2f}")
    print(f"  Std dev (last 100): {np.std(totals[-100:]):.2f}")
    wandb.finish()
"""


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
"""
# Train with random initialization (existing behavior)
python cartpole.py -a pid-learning --max-steps 300

# Train with human initialization (new feature)
python cartpole.py -a pid-learning --max-steps 300 --use-human-init fitted_pid_params.pkl

# Compare both in W&B
python cartpole.py -a pid-learning --max-steps 300 --run-name "Random Init Baseline"
python cartpole.py -a pid-learning --max-steps 300 --use-human-init fitted_pid_params.pkl --run-name "Human Init v1"
"""