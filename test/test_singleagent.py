import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from envs.HoverAviary import HoverAviary
from utils.logger import Logger
from utils.utils import sync
from envs.HoverAviary import HoverAviary
from envs.BaseSingleAgentAviary import ActionType, ObservationType


# Constants
DEFAULT_EXP_FOLDER = "results/hover-ppo"  # Default experiment folder
TEST_DURATION = 7  # Test duration in seconds
AGGR_PHY_STEPS = 5  # Physics steps per RL step
STARTING_POINT = np.array([[0, 0, 1.2]])  # Initial position

def find_latest_model():
    """Find and load the latest trained model."""
    exp_folders = [f for f in os.listdir("results") if f.startswith("hover-ppo")]
    if not exp_folders:
        raise FileNotFoundError("No matching experiment folders found in results/")
    
    latest_exp = sorted(exp_folders)[-1]
    path = os.path.join("results", latest_exp)
    
    model_files = {
        'best_model.zip': os.path.join(path, 'best_model.zip'),
        'success_model.zip': os.path.join(path, 'success_model.zip')
    }
    
    for file_name, file_path in model_files.items():
        if os.path.isfile(file_path):
            return PPO.load(file_path), path
    
    raise FileNotFoundError(f"No model found in {path}")

def run_test():
    """Run the test evaluation with visualization and logging."""
    # Load model and environment
    model, exp_path = find_latest_model()
    print(f"[INFO] Loaded model from: {exp_path}")
    
    # Create environments
    eval_env = HoverAviary(
        initial_xyzs=STARTING_POINT,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        obs=ObservationType.KIN,
        act=ActionType.RPM
    )
    
    test_env = HoverAviary(
        initial_xyzs=STARTING_POINT,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        gui=True,
        record=True
    )

    # Evaluate policy
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"\n[INFO] Evaluation results - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}\n")

    # Initialize logger
    logger = Logger(
        logging_freq_hz=int(test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS),
        num_drones=1,
        duration_sec=TEST_DURATION
    )

    # Run test
    obs = test_env.reset()
    start_time = time.time()
    
    for i in range(TEST_DURATION * int(test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        
        # Log data (state contains 20 elements as per BaseSingleAgentAviary)
        full_state = test_env._getDroneStateVector(0)
        logger.log(
            drone=0,
            timestamp=i/test_env.SIM_FREQ,
            state=full_state,  # حالت کامل 20 بعدی
            control=np.zeros(12)
        )
        
        # Sync simulation
        time_elapsed = time.time() - start_time
        expected_time = i * test_env.TIMESTEP
        if time_elapsed < expected_time:
            time.sleep(expected_time - time_elapsed)

    # Save results
    test_env.close()
    logger.save_as_csv("hover_test")
    logger.plot(pwm=False)  # Set pwm=True to plot PWM values instead of RPMs

if __name__ == "__main__":
    run_test()