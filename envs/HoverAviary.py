import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from gym import spaces
from envs.BaseAviary import DroneModel, Physics, BaseAviary
from envs.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary

class HoverAviary(BaseSingleAgentAviary):
    """Optimized single agent RL environment for CF2X drone hovering."""

    def __init__(self,
             initial_xyzs=None,
             initial_rpys=None,
             physics: Physics = Physics.PYB,
             reward_type: int = 0,
             freq: int = 240,
             aggregate_phy_steps: int = 1,
             gui=False,
             record=False,
             obs: ObservationType = ObservationType.KIN,
             act: ActionType = ActionType.RPM):
        """
        Initialize CF2X hovering environment.
        
        Args:
            initial_xyzs: Initial position (default: [0,0,1])
            initial_rpys: Initial orientation (default: [0,0,0])
            physics: Physics implementation (default: PYB)
            reward_type: Reward calculation type (default: 0)
            freq: Simulation frequency (Hz, default: 240)
            aggregate_phy_steps: Physics steps per control step (default: 1)
            gui: Show GUI (default: False)
            record: Record simulation (default: False)
            obs: Observation type (default: KIN)
            act: Action type (default: RPM)
        """
        super().__init__(
        initial_xyzs=initial_xyzs,
        initial_rpys=initial_rpys,
        physics=physics,
        freq=freq,
        aggregate_phy_steps=aggregate_phy_steps,
        gui=gui,
        record=record,
        obs=obs,
        act=act
    )
        self.reward_type = reward_type
        self._setup_reward_functions()

    def _setup_reward_functions(self):
        """Initialize reward calculation functions."""
        self._reward_functions = {
            0: self._reward_standard
        }

    def _reward_standard(self):
        """Standard reward function for hovering."""
        state = self._getDroneStateVector(0)
        dist_z = 0.9 * abs(1 - state[2])  # Penalize deviation from z=1
        dist_xy = 0.1 * (state[0]**2 + state[1]**2)  # Penalize deviation from origin
        return - (dist_z + dist_xy)  # Negative reward for distance from target

    def _computeReward(self):
        """Compute current reward value."""
        return self._reward_functions.get(self.reward_type, self._reward_standard)()

    def _computeDone(self):
        """Determine if episode is done."""
        return (self.step_counter / self.SIM_FREQ) > self.EPISODE_LEN_SEC

    def _computeInfo(self):
        """Return empty info dictionary."""
        return {}

    def _clipAndNormalizeState(self, state):
        """
        Normalize drone state to [-1,1] range.
        
        Args:
            state: Raw state vector (20 elements)
            
        Returns:
            Normalized state vector (20 elements)
        """
        # Define normalization bounds
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1
        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC
        MAX_PITCH_ROLL = np.pi

        # Clip values
        clipped = np.zeros_like(state)
        clipped[0:2] = np.clip(state[0:2], -MAX_XY, MAX_XY)  # XY position
        clipped[2] = np.clip(state[2], 0, MAX_Z)             # Z position
        clipped[7:9] = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)  # Roll/Pitch
        clipped[10:12] = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)  # XY velocity
        clipped[12] = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)          # Z velocity

        # Normalize values
        normalized = np.zeros_like(state)
        normalized[0:2] = clipped[0:2] / MAX_XY              # XY position
        normalized[2] = clipped[2] / MAX_Z                    # Z position
        normalized[3:7] = state[3:7]                          # Quaternion (no clip)
        normalized[7:9] = clipped[7:9] / MAX_PITCH_ROLL       # Roll/Pitch
        normalized[9] = state[9] / np.pi                      # Yaw
        normalized[10:12] = clipped[10:12] / MAX_LIN_VEL_XY   # XY velocity
        normalized[12] = clipped[12] / MAX_LIN_VEL_Z          # Z velocity
        
        # Angular velocity (normalize by magnitude)
        ang_vel_norm = np.linalg.norm(state[13:16])
        normalized[13:16] = state[13:16]/ang_vel_norm if ang_vel_norm != 0 else state[13:16]
        
        # RPM values (no normalization)
        normalized[16:20] = state[16:20]

        return normalized