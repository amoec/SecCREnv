"""
This file is an example train and test loop for the different environments.
Selecting different environments is done through setting the 'env_name' variable.

TODO:
* add rgb_array rendering for the different environments to allow saving videos
"""

import gymnasium as gym
from stable_baselines3 import SAC, TD3, DDPG, PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import numpy as np
import argparse
from typing import Union
import os
import yaml
import random
import csv

from SecCREnv.bluesky_gym import register_envs

# Register BlueSky environments
register_envs()

# Dictionary of supported algorithms
ALGORITHMS = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG,
    "DQN": DQN,
    "A2C": A2C
}
# Dictionary of off-policy algorithms
OFF_POLICY = {"SAC", "TD3", "DDPG", "DQN"}
    
class CSVLoggerCallback(BaseCallback):
    def __init__(self, log_dir, filename, verbose=0):
        super().__init__(verbose)
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)
        self.headers = [
            "timesteps",
            "episode",
            "reward"
        ]
        self.current_episode = 0
        self.buffer = []  # buffer for metrics
        self.flush_frequency = 1000  # flush every 1000 episodes
        self._write_header()
    
    def _write_header(self):
        if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
            with open(self.log_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
        
    def _flush_buffer(self):
        if self.buffer:
            with open(self.log_path, "a") as f:
                writer = csv.writer(f)
                for metrics in self.buffer:
                    writer.writerow([metrics[h] for h in self.headers])
            self.buffer = []
    
    def _on_step(self) -> bool:
        step_reward = self.locals.get("rewards", [0])[0]
        done = self.locals.get("dones", [False])[0]
        
        # Only increment episode count when an episode is done
        if done:
            self.current_episode += 1
        
        metrics = {
            "timesteps": self.num_timesteps,
            "episode": self.current_episode,
            "reward": step_reward
        }
        
        # Store metric in buffer
        self.buffer.append(metrics)
        
        # Flush buffer every `flush_frequency` episodes or if done on final steps (optional)
        if done and self.current_episode % self.flush_frequency == 0:
            self._flush_buffer()
                    
        return True
    
    def on_training_end(self):
        # Flush remaining buffer
        self._flush_buffer()

def load_algo_config(algo_name):
    """
    Load the YAML configuration for the specified algorithm.
    :param algo_name: Name of the RL algorithm (e.g., "PPO", "SAC").
    :return: Dictionary with `policy` and `init_args`.
    """
    config_path = f"atcenv_gym/atcenv/config/algos/{algo_name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file for {algo_name} not found: {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    if "policy" not in config['model'] or "init_args" not in config['model']:
        raise ValueError(f"Invalid configuration in {config_path}. Expected 'policy' and 'init_args' keys.")
    
    policy_type = config['model']["policy"]["type"]
    init_args = config['model']["init_args"]

    # Type conversion map
    type_conversions = {
        'learning_rate': float,
        'n_steps': int,
        'batch_size': int,
        'gamma': float,
        'gae_lambda': float,
        'ent_coef': lambda x: x if x == "auto" else float(x),
        'verbose': int,
        'buffer_size': int,
        'learning_starts': int,
        'tau': float,
        'train_freq': int,
        'gradient_steps': int,
        'target_update_interval': int,
        'exploration_fraction': float,
        'exploration_initial_eps': float,
        'exploration_final_eps': float
    }

    # Convert parameters to proper types
    for param, converter in type_conversions.items():
        if param in init_args:
            init_args[param] = converter(init_args[param])

    return policy_type, init_args

def main(args):
    env_name = 'SectorCREnv-v0'
    algorithm = args.algorithm
    train = args.train
    pretrain = float(args.pre_train) if args.pre_train != "full" else "full"
    eval_episodes = 10
    render_mode = "human" if args.render else None
    window = args.window
    
    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Define experiment folder
    lofi_experiment_folder = f"/scratch/amoec/ATC_RL/LoFi-{algorithm}/{algorithm}_{pretrain}"
    hifi_experiment_folder = f"/scratch/amoec/ATC_RL/HiFi-{algorithm}/{algorithm}_{pretrain}"
    
    # Load algorithm configuration from YAML
    policy_type, algo_params = load_algo_config(algorithm)

    # Initialize logger
    log_dir = f'{hifi_experiment_folder}/logs'
    os.makedirs(log_dir, exist_ok=True)
    file_name = 'results.csv'
    csv_logger_callback = CSVLoggerCallback(log_dir, file_name)
    
    # Initialize EarlyStopping callback
    callback_list = csv_logger_callback

    # Create environment
    env = gym.make(env_name, render_mode=None, seed=args.seed)

    # Initialize or load the model
    model_class = ALGORITHMS[algorithm]
    if pretrain != "full":
        # If no full training, then load the pre-trained model
        model_path = f"{lofi_experiment_folder}/model"
        model = model_class.load(model_path, env=env)
        print(f"Loaded pre-trained model from {model_path}")
        if algorithm in OFF_POLICY:
            rb_path = f"{model_path}_buffer"
            model.load_replay_buffer(rb_path)
            print(f"Replay buffer loaded from: {rb_path}")
    else:
        # Default model hyperparams
        model = model_class(policy_type, env)

    # Train the model
    if train:
        model.learn(total_timesteps=2e6, callback=callback_list)
        model_path = f"{hifi_experiment_folder}/model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        del model
        
    env.close()
    
    # Evaluate the trained model
    if args.eval:
        env = gym.make(env_name, render_mode=render_mode)
        for i in range(eval_episodes):
            done = truncated = False
            obs, info = env.reset()
            tot_rew = 0
            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action[()])
                tot_rew += reward
            print(f"Episode {i + 1}/{eval_episodes}: Total reward = {tot_rew}")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate RL models with BlueSky Gym.")
    parser.add_argument("--algorithm", type=str, choices=["SAC", "TD3", "DDPG", "PPO", "A2C", "DQN"], required=True, help="Algorithm to use.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model.")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation.")
    parser.add_argument("--pre_train", type=str, required=True, help="Pre-training percentage in LoFi env.")
    parser.add_argument("--window", type=int, required=True, help="Window size for the moving average.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    main(args)