import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from gym import spaces
import pybullet as p
import pybullet_data
from enum import Enum

from envs.BaseAviary import DroneModel, Physics, ImageType, BaseAviary
from utils.utils import nnlsRPM
from control.DSLPIDControl import DSLPIDControl

class ActionType(Enum):
    """Simplified action types for CF2X drone."""
    RPM = "rpm"    # Direct RPM control
    DYN = "dyn"    # Thrust and torque control
    PID = "pid"    # Position PID control
    VEL = "vel"    # Velocity control

class ObservationType(Enum):
    """Simplified observation types."""
    KIN = "kin"    # Kinematic state
    RGB = "rgb"    # RGB camera

class BaseSingleAgentAviary(BaseAviary):
    """Optimized single CF2X drone environment for RL."""

    def __init__(self,
             initial_xyzs=None,
             initial_rpys=None,
             physics: Physics = Physics.PYB,
             freq: int = 240,
             aggregate_phy_steps: int = 1,
             gui=False,
             record=False,
             obs: ObservationType = ObservationType.KIN,
             act: ActionType = ActionType.RPM):
        """
        Initialize single CF2X drone environment.
        
        Args:
            initial_xyzs: Initial position (3,)
            initial_rpys: Initial orientation (3,)
            physics: Physics implementation
            freq: Simulation frequency (Hz)
            aggregate_phy_steps: Physics steps per control step
            gui: Show GUI
            record: Record simulation
            obs: Observation type
            act: Action type
        """
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = 5
        
        # Initialize controller
        if act in [ActionType.PID, ActionType.VEL]:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            self.ctrl = DSLPIDControl()
            
        super().__init__(
        initial_xyzs=initial_xyzs if initial_xyzs is not None else np.array([[0, 0, 1]]),
        initial_rpys=initial_rpys if initial_rpys is not None else np.array([[0, 0, 0]]),
        physics=physics,
        freq=freq,
        aggregate_phy_steps=aggregate_phy_steps,
        gui=gui,
        record=record,
        obstacles=obs == ObservationType.RGB,
        user_debug_gui=False
    )
        
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000 / 3600)

    def _addObstacles(self):
        """Add obstacles only if using RGB observations."""
        if self.OBS_TYPE == ObservationType.RGB:
            obstacles = [
                ("block.urdf", [1, 0, 0.1]),
                ("cube_small.urdf", [0, 1, 0.1]),
                ("duck_vhacd.urdf", [-1, 0, 0.1]),
                ("teddy_vhacd.urdf", [0, -1, 0.1])
            ]
            for urdf, pos in obstacles:
                p.loadURDF(urdf, pos, p.getQuaternionFromEuler([0, 0, 0]), 
                          physicsClientId=self.CLIENT)

    def _actionSpace(self):
        """Get action space based on action type."""
        if self.ACT_TYPE in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE == ActionType.PID:
            size = 3
        else:
            raise ValueError(f"Unknown action type: {self.ACT_TYPE}")
            
        return spaces.Box(low=-1*np.ones(size), high=np.ones(size), dtype=np.float32)

    def _preprocessAction(self, action):
        """Convert action to RPMs based on action type."""
        if self.ACT_TYPE == ActionType.RPM:
            return np.array(self.HOVER_RPM * (1 + 0.05 * action))
            
        elif self.ACT_TYPE == ActionType.DYN:
            return nnlsRPM(
                thrust=(self.GRAVITY * (action[0] + 1)),
                x_torque=(0.05 * self.MAX_XY_TORQUE * action[1]),
                y_torque=(0.05 * self.MAX_XY_TORQUE * action[2]),
                z_torque=(0.05 * self.MAX_Z_TORQUE * action[3]),
                max_thrust=self.MAX_THRUST,
                max_xy_torque=self.MAX_XY_TORQUE,
                max_z_torque=self.MAX_Z_TORQUE,
                mixer_matrix=self.A,
                inv_mixer_matrix=self.INV_A,
                b_coeff=self.B_COEFF
            )
            
        elif self.ACT_TYPE == ActionType.PID:
            state = self._getDroneStateVector(0)
            rpm, _, _ = self.ctrl.computeControl(
                control_timestep=self.AGGR_PHY_STEPS * self.TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=state[0:3] + 0.1 * action
            )
            return rpm
            
        elif self.ACT_TYPE == ActionType.VEL:
            state = self._getDroneStateVector(0)
            v_unit_vector = np.zeros(3) if np.linalg.norm(action[0:3]) == 0 \
                else action[0:3] / np.linalg.norm(action[0:3])
                
            rpm, _, _ = self.ctrl.computeControl(
                control_timestep=self.AGGR_PHY_STEPS * self.TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=state[0:3],  # Maintain position
                target_rpy=np.array([0, 0, state[9]]),  # Maintain yaw
                target_vel=self.SPEED_LIMIT * np.abs(action[3]) * v_unit_vector
            )
            return rpm
            
        else:
            raise ValueError(f"Unknown action type: {self.ACT_TYPE}")

    def _observationSpace(self):
        """Get observation space based on observation type."""
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(
                low=0, high=255,
                shape=(self.IMG_RES[1], self.IMG_RES[0], 4),
                dtype=np.uint8
            )
        elif self.OBS_TYPE == ObservationType.KIN:
            return spaces.Box(
                low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
                high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown observation type: {self.OBS_TYPE}")

    def _computeObs(self):
        """Get current observation based on observation type."""
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
                self.rgb[0], _, _ = self._getDroneImages(0, segmentation=False)
                if self.RECORD:
                    self._exportImage(
                        img_type=ImageType.RGB,
                        img_input=self.rgb[0],
                        path=self.ONBOARD_IMG_PATH,
                        frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ)
                    )
            return self.rgb[0]
            
        elif self.OBS_TYPE == ObservationType.KIN:
            obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
            return np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            
        else:
            raise ValueError(f"Unknown observation type: {self.OBS_TYPE}")

    def _clipAndNormalizeState(self, state):
        """Normalize drone state to [-1,1] range (must be implemented by subclass)."""
        raise NotImplementedError