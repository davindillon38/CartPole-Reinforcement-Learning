#!/usr/bin/env python

import gym
import wandb
import numpy as np
import pickle
import argparse
from agent import AgentBasic, AgentRandom, AgentLearning
import stats
import random
import os
import glob

def environment_info(env):
    ''' Prints info about the given environment. '''
    print('************** Environment Info **************')
    print('Observation space: {}'.format(env.observation_space))
    print('Observation space high values: {}'.format(env.observation_space.high))
    print('Observation space low values: {}'.format(env.observation_space.low))
    print('Action space: {}'.format(env.action_space))
    print()


def validate_pid_consistency(Kp, Ki, Kd, Kx, n_trials=50, max_steps=5000):
    """
    Test if PID params consistently achieve high performance.
    
    Args:
        Kp, Ki, Kd, Kx: PID parameters to validate
        n_trials: Number of test episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary with performance statistics
    """
    print(f"\n{'='*70}")
    print(f"VALIDATING PID PARAMETERS")
    print(f"{'='*70}")
    print(f"Parameters: Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}, Kx={Kx:.4f}")
    print(f"Testing over {n_trials} trials with max_steps={max_steps}")
    print(f"{'-'*70}\n")
    
    episode_lengths = []
    env = gym.make('CartPole-v1')
    
    # Simple PID agent for validation
    class ValidationPID:
        def __init__(self, Kp, Ki, Kd, Kx):
            self.Kp = Kp
            self.Ki = Ki
            self.Kd = Kd
            self.Kx = Kx
            self.cumulative_theta = 0
            
        def reset(self):
            self.cumulative_theta = 0
            
        def act(self, obs):
            x, x_dot, theta, theta_dot = obs
            self.cumulative_theta += theta
            
            pid_value = (self.Kp * theta + 
                        self.Ki * self.cumulative_theta + 
                        self.Kd * theta_dot + 
                        self.Kx * x)
            
            return 1 if pid_value > 0 else 0
    
    agent = ValidationPID(Kp, Ki, Kd, Kx)
    
    for trial in range(n_trials):
        obs, _ = env.reset(seed=trial)
        agent.reset()
        
        total_steps = 0
        done = False
        
        while not done and total_steps < max_steps:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1
        
        episode_lengths.append(total_steps)
        
        # Progress indicator
        if (trial + 1) % 10 == 0:
            current_mean = np.mean(episode_lengths)
            print(f"Trial {trial+1}/{n_trials}: Current mean = {current_mean:.1f} steps")
    
    env.close()
    
    # Calculate statistics
    episode_lengths = np.array(episode_lengths)
    results = {
        'mean': np.mean(episode_lengths),
        'std': np.std(episode_lengths),
        'min': np.min(episode_lengths),
        'max': np.max(episode_lengths),
        'median': np.median(episode_lengths),
        'success_rate': np.sum(episode_lengths >= max_steps * 0.9) / n_trials,
        'perfect_rate': np.sum(episode_lengths >= max_steps) / n_trials,
        'episode_lengths': episode_lengths
    }
    
    # Print results
    print(f"\n{'='*70}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Mean:              {results['mean']:.2f} steps")
    print(f"Std Dev:           {results['std']:.2f} steps")
    print(f"Min:               {results['min']} steps")
    print(f"Max:               {results['max']} steps")
    print(f"Median:            {results['median']:.1f} steps")
    print(f"Success Rate (>90%): {results['success_rate']*100:.1f}%")
    print(f"Perfect Rate (100%): {results['perfect_rate']*100:.1f}%")
    print(f"{'='*70}\n")
    
    # Interpretation
    if results['mean'] >= max_steps * 0.95:
        print("✓ EXCELLENT: Parameters achieve near-perfect performance consistently")
    elif results['mean'] >= max_steps * 0.8:
        print("✓ GOOD: Parameters achieve strong performance")
    elif results['mean'] >= max_steps * 0.5:
        print("⚠ MODERATE: Parameters show moderate performance")
    else:
        print("✗ POOR: Parameters do not perform well - may need re-fitting")
    
    if results['std'] < max_steps * 0.1:
        print("✓ Low variance - very consistent performance")
    elif results['std'] < max_steps * 0.3:
        print("○ Moderate variance - reasonably consistent")
    else:
        print("⚠ High variance - inconsistent performance across trials")
    
    print()
    return results


def basic_guessing_policy(env, agent):
    ''' Execute basic agent policy. '''
    totals = []
    for episode in range(500):
        episode_rewards = 0
        obs, _ = env.reset()

        for step in range(1000):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_rewards += reward
            if done:
                break
        totals.append(episode_rewards)

    print('************** Reward Statistics **************')
    print('Average: {}'.format(np.mean(totals)))
    print('Standard Deviation: {}'.format(np.std(totals)))
    print('Minimum: {}'.format(np.min(totals)))
    print('Maximum: {}'.format(np.max(totals)))


def random_guessing_policy(env, agent):
    ''' Execute random agent policy. '''
    totals = []
    for episode in range(500):
        episode_rewards = 0
        obs, _ = env.reset()
        for step in range(1000):
            action = agent.act()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_rewards += reward
            if done:
                break
        totals.append(episode_rewards)

    print('Average: {}'.format(np.mean(totals)))
    print('Standard Deviation: {}'.format(np.std(totals)))
    print('Minimum: {}'.format(np.min(totals)))
    print('Maximum: {}'.format(np.max(totals)))


def q_learning(env, agent, episodes=2000):
    valid_actions = [0, 1]
    tolerance = 0.001
    training = True
    training_totals = []
    testing_totals = []
    history = {'epsilon': [], 'alpha': []}
    printed = False
    window_size = 100

    print(f'Starting Q-Learning for {episodes} episodes')
    print(f'Initial epsilon: {agent.epsilon:.4f}, alpha: {agent.alpha:.4f}\n')

    for episode in range(episodes):
        episode_rewards = 0
        obs, _ = env.reset()

        if agent.epsilon < tolerance:
            if not printed:
                print(f'\n{"="*60}')
                print(f'Training phase ended at episode {episode}')
                print(f'Epsilon dropped below {tolerance}')
                print(f'Starting testing phase (epsilon=0, alpha=0)')
                print(f'{"="*60}\n')
                printed = True
            agent.alpha = 0
            agent.epsilon = 0
            training = False

        agent.epsilon = agent.epsilon * 0.99975

        for step in range(200):
            state = agent.create_state(obs)
            agent.create_Q(state, valid_actions)
            action = agent.choose_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_rewards += reward

            if step != 0:
                agent.learn(state, action, prev_reward, prev_state, prev_action)
            prev_state = state
            prev_action = action
            prev_reward = reward
            if done:
                break

        if training:
            training_totals.append(episode_rewards)
            agent.training_trials += 1
            history['epsilon'].append(agent.epsilon)
            history['alpha'].append(agent.alpha)
            wandb.log({
                "training_reward": episode_rewards,
                "epsilon": agent.epsilon,
                "alpha": agent.alpha,
                "training_trials": agent.training_trials
            })
        else:
            testing_totals.append(episode_rewards)
            agent.testing_trials += 1
            wandb.log({
                "testing_reward": episode_rewards,
                "epsilon": agent.epsilon,
                "alpha": agent.alpha,
                "testing_trials": agent.testing_trials
            })
            if agent.testing_trials == 100:
                print(f'\nCompleted 100 testing episodes. Stopping.')
                break

        if (episode + 1) % 100 == 0:
            q_size = len(getattr(agent, 'Q', getattr(agent, 'q', getattr(agent, 'q_table', {}))))
            
            if training:
                recent_avg = np.mean(training_totals[-window_size:]) if len(training_totals) >= window_size else np.mean(training_totals)
                print(f'Episode {episode + 1}')
                print(f'  Phase: TRAINING')
                print(f'  Current reward: {episode_rewards:.1f}')
                print(f'  Recent avg: {recent_avg:.2f}')
                print(f'  Epsilon: {agent.epsilon:.6f}')
                print(f'  Alpha: {agent.alpha:.2f}')
                print(f'  Q-table size: {q_size}')
            else:
                recent_avg = np.mean(testing_totals[-window_size:]) if len(testing_totals) >= window_size else np.mean(testing_totals)
                print(f'Episode {episode + 1}')
                print(f'  Phase: TESTING')
                print(f'  Current reward: {episode_rewards:.1f}')
                print(f'  Testing avg: {recent_avg:.2f}')
                print(f'  Testing trials: {agent.testing_trials}')

    return training_totals, testing_totals, history


def get_next_run_number():
    os.makedirs("agents", exist_ok=True)
    existing_files = glob.glob("agents/pid_agent_run*_*.pkl")
    
    if not existing_files:
        return 1
    
    run_numbers = []
    for filename in existing_files:
        try:
            basename = os.path.basename(filename)
            parts = basename.split('_')
            if len(parts) >= 3 and parts[2].startswith('run'):
                run_num = int(parts[2][3:])
                run_numbers.append(run_num)
        except (ValueError, IndexError):
            continue
    
    return max(run_numbers) + 1 if run_numbers else 1


def save_q_learning_agent(agent, training_totals, testing_totals):
    os.makedirs("agents", exist_ok=True)
    
    if len(testing_totals) == 0:
        with open('CartPole-v0_stats.txt', 'w') as file_obj:
            file_obj.write('/-------- Q-Learning --------\\\n')
            file_obj.write('\n/---- Training Stats (Training Only - No Testing Phase) ----\\\n')
            file_obj.write('Average: {}\n'.format(np.mean(training_totals)))
            file_obj.write('Standard Deviation: {}\n'.format(np.std(training_totals)))
            file_obj.write('Minimum: {}\n'.format(np.min(training_totals)))
            file_obj.write('Maximum: {}\n'.format(np.max(training_totals)))
            file_obj.write('Number of training episodes: {}\n'.format(len(training_totals)))
            file_obj.write('\n/---- Testing Stats ----\\\n')
            file_obj.write('Testing phase not reached (epsilon did not drop below threshold)\n')
            file_obj.write('Final epsilon: {}\n'.format(agent.epsilon))
        
        agent_path = os.path.join('agents', 'q_learning_agent_training_only.pkl')
        with open(agent_path, 'wb') as f:
            pickle.dump(agent, f)
        
        print(f"\nStats saved to: CartPole-v0_stats.txt")
        print(f"Agent saved to: {agent_path}")
    else:
        stats.save_info(agent, training_totals, testing_totals)
        agent_path = os.path.join('agents', 'q_learning_agent_complete.pkl')
        with open(agent_path, 'wb') as f:
            pickle.dump(agent, f)
        print(f"\nAgent saved to: {agent_path}")


def load_human_pid_params(params_file):
    """Load human-fitted PID parameters from file."""
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Parameter file not found: {params_file}")
    
    with open(params_file, 'rb') as f:
        data = pickle.load(f)
    
    Kp, Ki, Kd, Kx = data['Kp'], data['Ki'], data['Kd'], data['Kx']
    
    print("\n" + "="*60)
    print("LOADED HUMAN-FITTED PID PARAMETERS")
    print("="*60)
    print(f"Kp = {Kp:.6f}")
    print(f"Ki = {Ki:.6f}")
    print(f"Kd = {Kd:.6f}")
    print(f"Kx = {Kx:.6f}")
    print("="*60 + "\n")
    
    return Kp, Ki, Kd, Kx


class AgentPIDLearning:
    def __init__(self, Kp=None, Ki=None, Kd=None, Kx=None, lr=0.01, max_steps=300):
        self.Kp = Kp if Kp is not None else np.random.uniform(0.5, 1.5)
        self.Ki = Ki if Ki is not None else np.random.uniform(0.0, 0.05)
        self.Kd = Kd if Kd is not None else np.random.uniform(0.5, 1.5)
        self.Kx = Kx if Kx is not None else np.random.uniform(0.1, 0.5)
        
        self.lr = lr
        self.max_steps = max_steps
        
        self.episode_history = []
        self.best_reward = 0
        self.best_params = (self.Kp, self.Ki, self.Kd, self.Kx)
        
    def reset_episode(self):
        self.cumulative_theta = 0
        self.episode_history = []

    def act(self, obs):
        x, x_dot, theta, theta_dot = obs
        
        self.cumulative_theta += theta
        
        pid_value = (self.Kp * theta + 
                    self.Ki * self.cumulative_theta + 
                    self.Kd * theta_dot +
                    self.Kx * x)
        
        self.episode_history.append({
            'obs': obs.copy(),
            'pid_value': pid_value,
            'x': x,
            'theta': theta,
            'theta_dot': theta_dot,
            'integral': self.cumulative_theta
        })
        
        action = 1 if pid_value > 0 else 0
        return action

    def learn_from_episode(self, total_reward):
        if len(self.episode_history) == 0:
            return
            
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_params = (self.Kp, self.Ki, self.Kd, self.Kx)
        
        avg_theta = np.mean([h['theta'] for h in self.episode_history])
        avg_theta_dot = np.mean([h['theta_dot'] for h in self.episode_history])
        avg_integral = np.mean([h['integral'] for h in self.episode_history])
        avg_x = np.mean([np.abs(h['x']) for h in self.episode_history])
        
        max_theta = np.max([np.abs(h['theta']) for h in self.episode_history])
        max_theta_dot = np.max([np.abs(h['theta_dot']) for h in self.episode_history])
        max_x = np.max([np.abs(h['x']) for h in self.episode_history])
        
        success_threshold = 0.65 * self.max_steps
        reward_factor = np.tanh(total_reward / success_threshold)
        
        performance_ratio = total_reward / self.max_steps
        
        if performance_ratio < 0.95:
            improvement_factor = (1 - performance_ratio) * (1 - reward_factor)
            
            self.Kp += self.lr * max_theta * improvement_factor * 2.0
            self.Kd += self.lr * max_theta_dot * improvement_factor * 2.0
            self.Ki += self.lr * np.abs(avg_integral) * 0.05 * improvement_factor
            self.Kx += self.lr * max_x * improvement_factor * 1.0
            
        else:
            noise_scale = 0.005 * (1 - performance_ratio)
            self.Kp += np.random.normal(0, max(noise_scale, 0.001))
            self.Ki += np.random.normal(0, max(noise_scale * 0.1, 0.0001))
            self.Kd += np.random.normal(0, max(noise_scale, 0.001))
            self.Kx += np.random.normal(0, max(noise_scale, 0.001))
        
        self.Kp = np.clip(self.Kp, 0.01, 50)
        self.Ki = np.clip(self.Ki, 0.0, 2.0)
        self.Kd = np.clip(self.Kd, 0.01, 50)
        self.Kx = np.clip(self.Kx, 0.0, 10)
        
        if performance_ratio < 0.98:
            self.lr *= 0.9999

    def save(self, filename="pid_agent.pkl"):
        if not filename.startswith("agents/"):
            filename = os.path.join("agents", filename)
        
        os.makedirs("agents", exist_ok=True)
        
        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"PID agent saved to {filename}")

    def load(self, filename="pid_agent.pkl"):
        if not os.path.exists(filename) and not filename.startswith("agents/"):
            filename = os.path.join("agents", filename)
            
        with open(filename, "rb") as f:
            self.__dict__ = pickle.load(f)
        print(f"PID agent loaded from {filename}")


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
    if run_number is None:
        run_number = get_next_run_number()
    
    env._max_episode_steps = max_steps
    
    if init_params is not None:
        Kp, Ki, Kd, Kx = init_params
        agent = AgentPIDLearning(Kp=Kp, Ki=Ki, Kd=Kd, Kx=Kx, lr=0.05, max_steps=max_steps)
        init_method = "HUMAN-INITIALIZED"
    else:
        agent = AgentPIDLearning(lr=0.05, max_steps=max_steps)
        init_method = "RANDOM"
    
    totals = []
    window_size = 100
    solved = False
    
    success_threshold = int(0.65 * max_steps)

    print("Starting PID Learning...")
    print(f"Initialization: {init_method}")
    print(f"Run number: {run_number}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Success threshold: {success_threshold} steps average")
    print(f"Initial params: Kp={agent.Kp:.4f}, Ki={agent.Ki:.5f}, Kd={agent.Kd:.4f}, Kx={agent.Kx:.4f}\n")
    
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
        
        agent.learn_from_episode(episode_reward)
        totals.append(episode_reward)
        
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

        if episode % 100 == 0:
            recent_avg = np.mean(totals[-window_size:]) if len(totals) >= window_size else np.mean(totals)
            print(f"Episode {episode}")
            print(f"  Current reward: {episode_reward:.1f}")
            print(f"  Recent avg ({min(len(totals), window_size)} eps): {recent_avg:.1f}")
            print(f"  Best reward: {agent.best_reward:.1f}")
            print(f"  Params: Kp={agent.Kp:.4f}, Ki={agent.Ki:.5f}, Kd={agent.Kd:.4f}, Kx={agent.Kx:.4f}")
            print(f"  Learning rate: {agent.lr:.6f}")
        
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

    suffix = "_human_init" if init_params is not None else ""
    final_filename = f"agents/pid_agent_run{run_number:03d}{suffix}_final.pkl"
    best_filename = f"agents/pid_agent_run{run_number:03d}{suffix}_best.pkl"
    
    os.makedirs("agents", exist_ok=True)
    
    agent.save(final_filename)
    
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent',
                        help='define type of agent you want (basic, random, q-learning, pid-learning, validate-pid)')
    parser.add_argument('-n', '--run_name', default="PID Learning",
                        help='Name of the W&B run')
    parser.add_argument('--max-steps', type=int, default=300,
                        help='Maximum steps per episode for PID learning (default: 300)')
    parser.add_argument('--episodes', type=int, default=17000,
                        help='Number of training episodes for q-learning (default: 17000)')
    parser.add_argument('--run-number', type=int, default=None,
                        help='Specific run number for saving files (auto-increments if not specified)')
    parser.add_argument('--use-human-init', type=str, default=None,
                        help='Path to human-fitted PID parameters file')
    parser.add_argument('--validate-params', type=str, default=None,
                        help='Path to PID parameters file to validate')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of validation trials (default: 50)')
    args = parser.parse_args()

    # For validation mode, don't initialize wandb
    if args.agent == 'validate-pid':
        if not args.validate_params:
            print("Error: --validate-params required for validate-pid mode")
            print("Usage: python cartpole.py -a validate-pid --validate-params fitted_pid_params.pkl --max-steps 5000")
            return
        
        try:
            Kp, Ki, Kd, Kx = load_human_pid_params(args.validate_params)
            validate_pid_consistency(Kp, Ki, Kd, Kx, n_trials=args.n_trials, max_steps=args.max_steps)
        except FileNotFoundError as e:
            print(f"Error: {e}")
        return

    wandb.init(
        project="cartpole-qlearning",
        entity="davindillon-ohio-university",
        name=args.run_name
    )

    env = gym.make('CartPole-v1')
    env.reset(seed=38)
    environment_info(env)

    if args.agent == 'basic':
        agent = AgentBasic()
        basic_guessing_policy(env, agent)
    elif args.agent == 'random':
        agent = AgentRandom(env.action_space)
        random_guessing_policy(env, agent)
    elif args.agent == 'q-learning':
        agent = AgentLearning(env, alpha=0.9, epsilon=1.0, gamma=0.9)
        training_totals, testing_totals, history = q_learning(env, agent, episodes=args.episodes)
        
        if len(testing_totals) > 0:
            stats.display_stats(agent, training_totals, testing_totals, history)
            save_q_learning_agent(agent, training_totals, testing_totals)
            if np.mean(testing_totals) >= 195.0:
                print("Environment SOLVED!!!")
            else:
                print("Environment not solved. Must get average reward of 195.0 or greater for 100 consecutive trials.")
        else:
            q_size = len(getattr(agent, 'Q', getattr(agent, 'q', getattr(agent, 'q_table', {}))))
            print("\n" + "="*60)
            print("Q-LEARNING TRAINING SUMMARY")
            print("="*60)
            print(f"Training episodes: {len(training_totals)}")
            print(f"Average reward (all): {np.mean(training_totals):.2f}")
            print(f"Average reward (last 100): {np.mean(training_totals[-100:]):.2f}")
            print(f"Best episode: {np.max(training_totals):.1f}")
            print(f"Final epsilon: {agent.epsilon:.6f}")
            print(f"Q-table size: {q_size}")
            print("\nNote: Did not reach testing phase (epsilon did not drop below tolerance)")
            print("Consider running more episodes or adjusting epsilon decay rate.")
            print("="*60)
            
            save_q_learning_agent(agent, training_totals, testing_totals)
        
        wandb.finish()
    elif args.agent == 'pid-learning':
        init_params = None
        if args.use_human_init:
            try:
                init_params = load_human_pid_params(args.use_human_init)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("Falling back to random initialization")
        
        totals = pid_learning(env, episodes=args.episodes if hasattr(args, 'episodes') else 10000,
                             max_steps=args.max_steps, run_number=args.run_number,
                             init_params=init_params)
        print(f"\nFinal Statistics:")
        print(f"  Average reward (all episodes): {np.mean(totals):.2f}")
        print(f"  Average reward (last 100): {np.mean(totals[-100:]):.2f}")
        print(f"  Std dev (last 100): {np.std(totals[-100:]):.2f}")
        wandb.finish()
    else:
        agent = AgentBasic()
        basic_guessing_policy(env, agent)


if __name__ == '__main__':
    main()
