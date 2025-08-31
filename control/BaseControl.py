import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import numpy as np
import pybullet as p
from enum import Enum
import xml.etree.ElementTree as etxml
from scipy.spatial.transform import Rotation

from envs.BaseAviary import DroneModel, BaseAviary

class BaseControl(object):
    """Base class for control of CF2X drone model.

    Implements basic control interface and utilities for CF2X drone.
    """

    ################################################################################

    def __init__(self, g: float=9.8):
        
        
        self.DRONE_MODEL = "cf2x"  
        self.GRAVITY = g * self._getURDFParameter('m')
        self.KF = self._getURDFParameter('kf')  # Thrust coefficient
        self.KM = self._getURDFParameter('km')  # Torque coefficient
        self.reset()

    ################################################################################

    def reset(self):
        """Reset the control counter."""
        self.control_counter = 0

    ################################################################################

    def computeControlFromState(self,
                              control_timestep,
                              state,
                              target_pos,
                              target_rpy=np.zeros(3),
                              target_vel=np.zeros(3),
                              target_rpy_rates=np.zeros(3)):
        """Compute control from current state.
        """
        return self.computeControl(
            control_timestep=control_timestep,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=target_vel,
            target_rpy_rates=target_rpy_rates
        )

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
        """Abstract method to compute control action.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    ################################################################################
    
    def _getURDFParameter(self, parameter_name: str):
        """Get parameter from CF2X URDF file."""
        URDF = "cf2x.urdf"  
        URDF_TREE = etxml.parse(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+URDF).getroot()
        
        if parameter_name == 'm':
            return float(URDF_TREE[1][0][1].attrib['value'])
        elif parameter_name in ['ixx', 'iyy', 'izz']:
            return float(URDF_TREE[1][0][2].attrib[parameter_name])
        elif parameter_name in ['arm', 'thrust2weight', 'kf', 'km', 'max_speed_kmh', 
                              'gnd_eff_coeff', 'prop_radius', 'drag_coeff_xy', 
                              'drag_coeff_z']:
            return float(URDF_TREE[0].attrib[parameter_name])
        elif parameter_name in ['length', 'radius']:
            return float(URDF_TREE[1][2][1][0].attrib[parameter_name])
        elif parameter_name == 'collision_z_offset':
            COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
            return COLLISION_SHAPE_OFFSETS[2]
        else:
            raise ValueError(f"Unknown parameter: {parameter_name}")

    ################################################################################

    def setPIDCoefficients(self,
                          p_coeff_pos=None,
                          i_coeff_pos=None,
                          d_coeff_pos=None,
                          p_coeff_att=None,
                          i_coeff_att=None,
                          d_coeff_att=None):
        """Set PID coefficients if using PID control."""
        if not all(hasattr(self, attr) for attr in ['P_COEFF_FOR', 'I_COEFF_FOR', 'D_COEFF_FOR',
                                                  'P_COEFF_TOR', 'I_COEFF_TOR', 'D_COEFF_TOR']):
            raise AttributeError("PID coefficients not initialized in this control class")
            
        if p_coeff_pos is not None:
            self.P_COEFF_FOR = p_coeff_pos
        if i_coeff_pos is not None:
            self.I_COEFF_FOR = i_coeff_pos
        if d_coeff_pos is not None:
            self.D_COEFF_FOR = d_coeff_pos
        if p_coeff_att is not None:
            self.P_COEFF_TOR = p_coeff_att
        if i_coeff_att is not None:
            self.I_COEFF_TOR = i_coeff_att
        if d_coeff_att is not None:
            self.D_COEFF_TOR = d_coeff_att