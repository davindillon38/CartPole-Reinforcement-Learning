#!/usr/bin/env python
"""
Gamepad demonstration collector for CartPole.
Records human gameplay with Logitech F310 controller.
Uses custom rendering to avoid pygame event conflicts.
"""

import pygame
import gym
import numpy as np
import pickle
import json
import os
from datetime import datetime
import math


class CartPoleRenderer:
    """Custom CartPole renderer that doesn't interfere with joystick."""
    
    def __init__(self):
        self.screen_width = 600
        self.screen_height = 400
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("CartPole - Human Control")
        self.clock = pygame.time.Clock()
        
        # CartPole rendering constants
        self.world_width = 4.8  # CartPole world is 4.8 units wide
        self.scale = self.screen_width / self.world_width
        self.cart_y = self.screen_height / 2
        self.cart_width = 50
        self.cart_height = 30
        self.pole_width = 10
        self.pole_len = 100  # Visual length
        
        # Colors
        self.bg_color = (240, 240, 240)
        self.cart_color = (50, 50, 200)
        self.pole_color = (200, 50, 50)
        self.track_color = (100, 100, 100)
        self.text_color = (0, 0, 0)
        
        # Font
        self.font = pygame.font.Font(None, 24)
    
    def render(self, obs, step, total_reward, action):
        """
        Render the CartPole state.
        
        Args:
            obs: [x, x_dot, theta, theta_dot]
            step: Current step number
            total_reward: Total reward so far
            action: Current action (0=left, 1=right)
        """
        x, x_dot, theta, theta_dot = obs
        
        self.screen.fill(self.bg_color)
        
        # Draw track
        track_y = int(self.cart_y + self.cart_height / 2)
        pygame.draw.line(self.screen, self.track_color, 
                        (0, track_y), (self.screen_width, track_y), 3)
        
        # Convert world coordinates to screen coordinates
        cart_x = int((x + self.world_width / 2) * self.scale)
        
        # Draw cart
        cart_rect = pygame.Rect(
            cart_x - self.cart_width / 2,
            self.cart_y - self.cart_height / 2,
            self.cart_width,
            self.cart_height
        )
        pygame.draw.rect(self.screen, self.cart_color, cart_rect)
        
        # Draw pole
        pole_end_x = cart_x + math.sin(theta) * self.pole_len
        pole_end_y = self.cart_y - math.cos(theta) * self.pole_len
        pygame.draw.line(self.screen, self.pole_color,
                        (cart_x, self.cart_y),
                        (pole_end_x, pole_end_y),
                        self.pole_width)
        
        # Draw pole tip (circle)
        pygame.draw.circle(self.screen, self.pole_color,
                          (int(pole_end_x), int(pole_end_y)), 8)
        
        # Draw action indicator
        action_text = "◀ LEFT" if action == 0 else "RIGHT ▶"
        action_color = (255, 100, 100) if action == 0 else (100, 255, 100)
        action_surf = self.font.render(action_text, True, action_color)
        self.screen.blit(action_surf, (10, 10))
        
        # Draw stats
        stats_text = f"Step: {step}  |  Reward: {total_reward:.0f}"
        stats_surf = self.font.render(stats_text, True, self.text_color)
        self.screen.blit(stats_surf, (10, 40))
        
        # Draw angle indicator
        angle_deg = theta * 180 / math.pi
        angle_text = f"Angle: {angle_deg:+.1f}°"
        angle_surf = self.font.render(angle_text, True, self.text_color)
        self.screen.blit(angle_surf, (10, 70))
        
        # Draw position indicator
        pos_text = f"Position: {x:+.2f}"
        pos_surf = self.font.render(pos_text, True, self.text_color)
        self.screen.blit(pos_surf, (10, 100))
        
        # Draw controls reminder
        controls = "D-pad: LEFT/RIGHT  |  A: Skip  |  X: Save & Exit"
        controls_surf = self.font.render(controls, True, (100, 100, 100))
        self.screen.blit(controls_surf, (10, self.screen_height - 30))
        
        pygame.display.flip()


class GamepadController:
    """Interface for Logitech F310 gamepad."""
    
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected! Please connect your Logitech F310.")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        print(f"Gamepad connected: {self.joystick.get_name()}")
        print("Controls:")
        print("  D-pad LEFT  = Push cart left")
        print("  D-pad RIGHT = Push cart right")
        print("  A button    = Skip current episode")
        print("  X button    = Save and exit")
    
    def cleanup(self):
        pygame.quit()


class DemonstrationCollector:
    """Collects and saves human demonstrations."""
    
    def __init__(self, env_name='CartPole-v1', fps=10):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.fps = fps
        self.frame_delay = int(1000 / fps)  # Convert FPS to milliseconds
        
        self.demonstrations = []
        self.controller = GamepadController()
        self.renderer = CartPoleRenderer()
        
        # Create demos directory
        os.makedirs('demos', exist_ok=True)
        
        print(f"\nRunning at {fps} FPS ({self.frame_delay}ms per frame)")
        print("Tip: Use --fps flag to adjust speed (e.g., --fps 8 for slower)\n")
    
    def collect_episode(self, episode_num):
        """Collect a single episode of human demonstration."""
        obs, _ = self.env.reset()
        
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'episode_num': episode_num,
            'timestamp': datetime.now().isoformat()
        }
        
        total_reward = 0
        step = 0
        last_action = 1  # Default to right if no input
        
        print(f"\n=== Episode {episode_num} ===")
        print("Use D-pad to control the cart. Episode starts in 1 second...")
        pygame.time.wait(1000)
        
        done = False
        while not done:
            # Process pygame events to update joystick state
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None, True
            
            # Read joystick directly
            hat = self.controller.joystick.get_hat(0)
            a_button = self.controller.joystick.get_button(0)
            x_button = self.controller.joystick.get_button(2)
            
            # Determine action from D-pad
            action = None
            if hat[0] == -1:  # D-pad LEFT
                action = 0
            elif hat[0] == 1:  # D-pad RIGHT
                action = 1
            
            # Handle save and exit
            if x_button:
                print("\nX button pressed - saving and exiting...")
                return None, True
            
            # Handle quit episode
            if a_button:
                print("\nA button pressed - ending episode early")
                break
            
            # If no input, use last action
            if action is None:
                action = last_action
            else:
                last_action = action
            
            # Render current state
            self.renderer.render(obs, step, total_reward, action)
            
            # Store state before action
            episode_data['observations'].append(obs.copy())
            episode_data['actions'].append(action)
            
            # Take action
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            episode_data['rewards'].append(reward)
            total_reward += reward
            step += 1
            
            # Debug: print controller state
            if step % 5 == 0:  # Print every 5 steps to avoid spam
                print(f"Step {step}: Hat={hat}, action={action} ({'LEFT' if action == 0 else 'RIGHT'}), reward={total_reward:.0f}")
            
            # Slow down for human reaction time
            pygame.time.wait(self.frame_delay)
        
        episode_data['total_reward'] = total_reward
        episode_data['steps'] = step
        
        print(f"Episode finished: {step} steps, reward = {total_reward}")
        
        return episode_data, False
    
    def collect_demonstrations(self, num_episodes=10):
        """Collect multiple episodes of demonstrations."""
        print("\n" + "="*60)
        print("HUMAN DEMONSTRATION COLLECTION")
        print("="*60)
        print(f"Target: {num_episodes} episodes")
        print("\nMake sure your Logitech F310 switch is set to 'D' mode")
        print("Try to balance the pole as long as possible!")
        print("\nPress any D-pad direction to start...")
        print("="*60)
        
        # Wait for initial input
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return []
            hat = self.controller.joystick.get_hat(0)
            if hat[0] != 0:
                waiting = False
            pygame.time.wait(50)
        
        episode_count = 0
        while episode_count < num_episodes:
            episode_data, should_exit = self.collect_episode(episode_count + 1)
            
            if should_exit:
                break
            
            if episode_data is not None:
                self.demonstrations.append(episode_data)
                episode_count += 1
                
                # Show progress
                if episode_count < num_episodes:
                    print(f"\n{episode_count}/{num_episodes} episodes complete")
                    print("Get ready for next episode...")
                    pygame.time.wait(2000)
        
        self.save_demonstrations()
        self.print_summary()
        
        return self.demonstrations
    
    def save_demonstrations(self):
        """Save demonstrations to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as pickle
        pkl_filename = f"demos/human_demos_{timestamp}.pkl"
        with open(pkl_filename, 'wb') as f:
            pickle.dump(self.demonstrations, f)
        
        # Save metadata as JSON
        metadata = {
            'num_episodes': len(self.demonstrations),
            'timestamp': timestamp,
            'env_name': self.env_name,
            'total_steps': sum(d['steps'] for d in self.demonstrations),
            'avg_reward': np.mean([d['total_reward'] for d in self.demonstrations]) if self.demonstrations else 0,
            'episodes': [
                {
                    'episode_num': d['episode_num'],
                    'steps': d['steps'],
                    'reward': d['total_reward']
                }
                for d in self.demonstrations
            ]
        }
        
        json_filename = f"demos/human_demos_{timestamp}_metadata.json"
        with open(json_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDemonstrations saved:")
        print(f"  - {pkl_filename}")
        print(f"  - {json_filename}")
    
    def print_summary(self):
        """Print summary statistics of collected demonstrations."""
        if not self.demonstrations:
            print("\nNo demonstrations collected.")
            return
        
        rewards = [d['total_reward'] for d in self.demonstrations]
        steps = [d['steps'] for d in self.demonstrations]
        
        print("\n" + "="*60)
        print("DEMONSTRATION SUMMARY")
        print("="*60)
        print(f"Episodes collected: {len(self.demonstrations)}")
        print(f"Total steps: {sum(steps)}")
        print(f"\nReward statistics:")
        print(f"  Mean: {np.mean(rewards):.2f}")
        print(f"  Std:  {np.std(rewards):.2f}")
        print(f"  Min:  {np.min(rewards):.2f}")
        print(f"  Max:  {np.max(rewards):.2f}")
        print(f"\nSteps statistics:")
        print(f"  Mean: {np.mean(steps):.2f}")
        print(f"  Std:  {np.std(steps):.2f}")
        print(f"  Min:  {np.min(steps):.2f}")
        print(f"  Max:  {np.max(steps):.2f}")
        print("="*60)
    
    def cleanup(self):
        """Clean up resources."""
        self.controller.cleanup()
        self.env.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect human demonstrations for CartPole')
    parser.add_argument('-n', '--num-episodes', type=int, default=10,
                        help='Number of episodes to collect (default: 10)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second (default: 10, lower is slower/easier)')
    args = parser.parse_args()
    
    collector = None
    try:
        collector = DemonstrationCollector(fps=args.fps)
        collector.collect_demonstrations(num_episodes=args.num_episodes)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if collector is not None:
            collector.cleanup()
        print("\nGoodbye!")


if __name__ == '__main__':
    main()
