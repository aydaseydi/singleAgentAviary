import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from control.BaseControl import BaseControl

class DSLPIDControl(BaseControl):
    """PID control class for CF2X drones based on UTIAS' DSL work."""

    ################################################################################

    def __init__(self, g: float=9.8):
        """Initialize CF2X PID control."""
        super().__init__(g=g)
        
        # PID coefficients for CF2X
        self.P_COEFF_FOR = np.array([.4, .4, 1.25])      # Position
        self.I_COEFF_FOR = np.array([.05, .05, .05])     # Position
        self.D_COEFF_FOR = np.array([.2, .2, .5])        # Position
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])  # Attitude
        self.I_COEFF_TOR = np.array([.0, .0, 500.])      # Attitude
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])  # Attitude
        
        # PWM to RPM conversion constants for CF2X
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        
        # CF2X-specific mixer matrix
        self.MIXER_MATRIX = np.array([
            [.5, -.5, -1], 
            [.5, .5, 1], 
            [-.5, .5, -1], 
            [-.5, -.5, 1]
        ])
        
        self.reset()

    ################################################################################

    def reset(self):
        """Reset control variables."""
        super().reset()
        self.last_rpy = np.zeros(3)
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    ################################################################################
    
    def computeControl(self,
                      control_timestep,
                      cur_pos,
                      cur_quat,
                      cur_vel,
                      cur_ang_vel,
                      target_pos,
                      target_rpy=np.zeros(3),
                      target_vel=np.zeros(3),
                      target_rpy_rates=np.zeros(3)):
        """Compute PID control action for CF2X drone."""
        self.control_counter += 1
        
        # Position control
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(
            control_timestep,
            cur_pos,
            cur_quat,
            cur_vel,
            target_pos,
            target_rpy,
            target_vel
        )
        
        # Attitude control
        rpm = self._dslPIDAttitudeControl(
            control_timestep,
            thrust,
            cur_quat,
            computed_target_rpy,
            target_rpy_rates
        )
        
        # Calculate yaw error
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
    
    ################################################################################

    def _dslPIDPositionControl(self,
                             control_timestep,
                             cur_pos,
                             cur_quat,
                             cur_vel,
                             target_pos,
                             target_rpy,
                             target_vel):
        """PID position control for CF2X."""
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        
        # Update integral terms with clamping
        self.integral_pos_e += pos_e * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
        
        # Calculate target thrust using PID
        target_thrust = (self.P_COEFF_FOR * pos_e + 
                        self.I_COEFF_FOR * self.integral_pos_e + 
                        self.D_COEFF_FOR * vel_e + 
                        np.array([0, 0, self.GRAVITY]))
        
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        
        # Calculate target orientation
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = np.vstack([target_x_ax, target_y_ax, target_z_ax]).T
        
        target_euler = Rotation.from_matrix(target_rotation).as_euler('XYZ', degrees=False)
        return thrust, target_euler, pos_e
    
    ################################################################################

    def _dslPIDAttitudeControl(self,
                             control_timestep,
                             thrust,
                             cur_quat,
                             target_euler,
                             target_rpy_rates):
        """PID attitude control for CF2X."""
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        
        # Calculate rotation error
        target_quat = Rotation.from_euler('XYZ', target_euler, degrees=False).as_quat()
        target_rotation = Rotation.from_quat(target_quat).as_matrix()
        rot_matrix_e = target_rotation.T @ cur_rotation - cur_rotation.T @ target_rotation
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        
        # Calculate angular rate error
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        
        # Update integral terms with clamping
        self.integral_rpy_e -= rot_e * control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[:2] = np.clip(self.integral_rpy_e[:2], -1., 1.)
        
        # Calculate target torques using PID
        target_torques = (-self.P_COEFF_TOR * rot_e + 
                         self.D_COEFF_TOR * rpy_rates_e + 
                         self.I_COEFF_TOR * self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        
        # Convert to PWM and then to RPM
        pwm = thrust + self.MIXER_MATRIX @ target_torques
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST