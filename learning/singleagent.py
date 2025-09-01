import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from envs.HoverAviary import HoverAviary
from envs.BaseSingleAgentAviary import ActionType, ObservationType

# Constants
EPISODE_REWARD_THRESHOLD = -0
AGGR_PHY_STEPS = 5
TOTAL_TIMESTEPS = 1000000
EVAL_FREQ = 2000

def train():
    """Main training function with default parameters."""
    # Create log directory
    timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    log_dir = f"results/hover-ppo-{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment (removed drone_model parameter)
    train_env = HoverAviary(
        initial_xyzs=np.array([[0, 0, 1]]),
        aggregate_phy_steps=AGGR_PHY_STEPS,
        obs=ObservationType.KIN,
        act=ActionType.RPM
    )
    
    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)

    # Create model
    policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=[
        dict(shared=[128, 128], pi=[128, 256], vf=[256])
    ]
)
    
    model = PPO(
    "MlpPolicy",
    train_env,
    policy_kwargs=policy_kwargs,
    tensorboard_log=os.path.join("logs", "ppo_quad"),
    verbose=1,
    device='auto'  # Let PyTorch choose the best device
)

    # Setup callbacks
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=EPISODE_REWARD_THRESHOLD,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        train_env,
        callback_on_new_best=stop_callback,
        verbose=1,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False
    )

    # Train model
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        log_interval=100,
    )

    # Save final model
    model.save(os.path.join(log_dir, 'success_model.zip'))
    print(f"[INFO] Training complete. Model saved to: {log_dir}")

if __name__ == "__main__":
    train()