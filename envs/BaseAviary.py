import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sys import platform
import time
import collections
from datetime import datetime
from enum import Enum, auto
import xml.etree.ElementTree as etxml
from PIL import Image
import numpy as np
import pybullet as p
import pybullet_data
import gym

class DroneModel(Enum):
    """Supported drone models (optimized for CF2X)."""
    CF2X = auto()  # Bitcraze Crazyflie 2.0 in X configuration
    
    @classmethod
    def get_cf2x(cls):
        """Get the CF2X model instance."""
        return cls.CF2X

################################################################################

class Physics(Enum):
    """Physics simulation modes."""
    PYB = auto()            # Base PyBullet physics
    PYB_GND = auto()        # With ground effect
    PYB_DRAG = auto()       # With aerodynamic drag
    PYB_DW = auto()         # With downwash
    PYB_GND_DRAG_DW = auto() # With all effects
    DYN = auto()            # Explicit dynamics
    
################################################################################

class ImageType(Enum):
    """Camera image types."""
    RGB = auto()  # Color image (RGBA)
    DEPTH = auto()  # Depth map
    SEG = auto()  # Segmentation

################################################################################

class BaseAviary(gym.Env):
    """Base class for implementing a single CF2X drone environment"""

    metadata = {'render.modes': ['human']}
    
    ################################################################################

    def __init__(self,
             drone_model: DroneModel = DroneModel.CF2X,
             initial_xyzs=None,  # اضافه کردن این پارامتر
             initial_rpys=None,  # اضافه کردن این پارامتر
             physics: Physics = Physics.PYB,
             freq: int = 240,
             aggregate_phy_steps: int = 1,
             gui=False,
             record=False,
             obstacles=False,
             user_debug_gui=True):
        '''Constant parameters'''
        self.G = 9.8
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.SIM_FREQ = freq
        self.TIMESTEP = 1./self.SIM_FREQ
        self.AGGR_PHY_STEPS = aggregate_phy_steps
        
        self.NUM_DRONES = 1  # Only single drone supported
        self.DRONE_MODEL = drone_model
        assert self.DRONE_MODEL == DroneModel.CF2X, "Only CF2X drone model is supported"
        
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = physics
        self.OBSTACLES = obstacles
        self.USER_DEBUG = user_debug_gui
        self.URDF = "cf2x.urdf"
        
        '''Extract drone parameters from URDF'''
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()
        
        print("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
            self.M, self.L, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        
        #### Compute max RPM, thrust and torque #####################################
        self.GRAVITY = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        
        '''PyBullet setup'''
        if self.GUI:
            self.CLIENT = p.connect(p.GUI)
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=1.45,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 0.5],
                                         physicsClientId=self.CLIENT)
            if self.USER_DEBUG:
                self.SLIDERS = -1*np.ones(4)
                for i in range(4):
                    self.SLIDERS[i] = p.addUserDebugParameter("Propeller "+str(i)+" RPM", 0, self.MAX_RPM, self.HOVER_RPM, physicsClientId=self.CLIENT)
                self.INPUT_SWITCH = p.addUserDebugParameter("Use GUI RPM", 9999, -1, 0, physicsClientId=self.CLIENT)
        else:
            self.CLIENT = p.connect(p.DIRECT)
            if self.RECORD:
                self.VID_WIDTH = int(640)
                self.VID_HEIGHT = int(480)
                self.FRAME_PER_SEC = 24
                self.CAPTURE_FREQ = int(self.SIM_FREQ/self.FRAME_PER_SEC)
                self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(distance=2.8,
                                                                    yaw=-30,
                                                                    pitch=-30,
                                                                    roll=0,
                                                                    cameraTargetPosition=[0, 0, 1],
                                                                    upAxisIndex=2,
                                                                    physicsClientId=self.CLIENT)
                self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                            aspect=self.VID_WIDTH/self.VID_HEIGHT,
                                                            nearVal=0.1,
                                                            farVal=1000.0)
        
        # Initial position and orientation
        if initial_xyzs is None:
            self.INIT_XYZS = np.array([[0, 0, self.COLLISION_H/2-self.COLLISION_Z_OFFSET+.1]])
        else:
            self.INIT_XYZS = np.array(initial_xyzs).reshape(1, 3)
            
        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((1, 3))
        else:
            self.INIT_RPYS = np.array(initial_rpys).reshape(1, 3)
        
        # Create action and observation spaces
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        
        # Initialize the environment
        self._housekeeping()
        self._updateAndStoreKinematicInformation()
        self._startVideoRecording()
    
    ################################################################################

    def reset(self):
        """Reset the environment"""
        p.resetSimulation(physicsClientId=self.CLIENT)
        self._housekeeping()
        self._updateAndStoreKinematicInformation()
        self._startVideoRecording()
        return self._computeObs()
    
    ################################################################################

    def step(self, action):
        """Advance the simulation by one step"""
        # Record video frames if needed
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                     height=self.VID_HEIGHT,
                                                     shadow=1,
                                                     viewMatrix=self.CAM_VIEW,
                                                     projectionMatrix=self.CAM_PRO,
                                                     renderer=p.ER_TINY_RENDERER,
                                                     physicsClientId=self.CLIENT)
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
            
        # Read input from GUI if enabled
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = not self.USE_GUI_RPM
                
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.array([self.gui_input])
            if self.step_counter%(self.SIM_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = p.addUserDebugText("Using GUI RPM",
                                                         textPosition=[0, 0, 0],
                                                         textColorRGB=[1, 0, 0],
                                                         lifeTime=1,
                                                         textSize=2,
                                                         parentObjectUniqueId=self.DRONE_IDS[0],
                                                         parentLinkIndex=-1,
                                                         replaceItemUniqueId=int(self.GUI_INPUT_TEXT),
                                                         physicsClientId=self.CLIENT)
        else:
            self._saveLastAction(action)
            clipped_action = np.reshape(self._preprocessAction(action), (1, 4))
            
        # Apply physics for each aggregation step
        for _ in range(self.AGGR_PHY_STEPS):
            if self.AGGR_PHY_STEPS > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
                
            if self.PHYSICS == Physics.PYB:
                self._physics(clipped_action[0, :], 0)
            elif self.PHYSICS == Physics.DYN:
                self._dynamics(clipped_action[0, :], 0)
            elif self.PHYSICS == Physics.PYB_GND:
                self._physics(clipped_action[0, :], 0)
                self._groundEffect(clipped_action[0, :], 0)
            elif self.PHYSICS == Physics.PYB_DRAG:
                self._physics(clipped_action[0, :], 0)
                self._drag(self.last_clipped_action[0, :], 0)
            elif self.PHYSICS == Physics.PYB_DW:
                self._physics(clipped_action[0, :], 0)
                self._downwash(0)
            elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                self._physics(clipped_action[0, :], 0)
                self._groundEffect(clipped_action[0, :], 0)
                self._drag(self.last_clipped_action[0, :], 0)
                self._downwash(0)
                
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
                
            self.last_clipped_action = clipped_action
        
        self._updateAndStoreKinematicInformation()
        obs = self._computeObs()
        reward = self._computeReward()
        done = self._computeDone()
        info = self._computeInfo()

        self.step_counter += (1 * self.AGGR_PHY_STEPS)
        return obs, reward, done, info
    
    ################################################################################
    
    def render(self, mode='human', close=False):
        """Render the environment state"""
        if self.first_render_call and not self.GUI:
            print("[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface")
            self.first_render_call = False
            
        print("\n[INFO] BaseAviary.render() ——— it {:04d}".format(self.step_counter),
              "——— wall-clock time {:.1f}s,".format(time.time()-self.RESET_TIME),
              "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(self.step_counter*self.TIMESTEP, self.SIM_FREQ, (self.step_counter*self.TIMESTEP)/(time.time()-self.RESET_TIME)))
        
        print("[INFO] BaseAviary.render() ——— drone 0",
              "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(self.pos[0, 0], self.pos[0, 1], self.pos[0, 2]),
              "——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.vel[0, 0], self.vel[0, 1], self.vel[0, 2]),
              "——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.rpy[0, 0]*self.RAD2DEG, self.rpy[0, 1]*self.RAD2DEG, self.rpy[0, 2]*self.RAD2DEG),
              "——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— ".format(self.ang_v[0, 0], self.ang_v[0, 1], self.ang_v[0, 2]))
    
    ################################################################################

    def close(self):
        """Close the PyBullet environment"""
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.CLIENT)
        p.disconnect(physicsClientId=self.CLIENT)
    
    ################################################################################

    def getPyBulletClient(self):
        """Get the PyBullet client ID"""
        return self.CLIENT
    
    ################################################################################

    def getDroneIds(self):
        """Get the drone's PyBullet ID"""
        return self.DRONE_IDS
    
    ################################################################################

    def _housekeeping(self):
        """Initialize/reset internal variables"""
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True
        self.X_AX = -1
        self.Y_AX = -1
        self.Z_AX = -1
        self.GUI_INPUT_TEXT = -1
        self.USE_GUI_RPM = False
        self.last_input_switch = 0
        self.last_action = -1*np.ones((1, 4))
        self.last_clipped_action = np.zeros((1, 4))
        self.gui_input = np.zeros(4)
        
        # Kinematic arrays
        self.pos = np.zeros((1, 3))
        self.quat = np.zeros((1, 4))
        self.rpy = np.zeros((1, 3))
        self.vel = np.zeros((1, 3))
        self.ang_v = np.zeros((1, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((1, 3))
            
        # PyBullet setup
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        
        # Load models
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        self.DRONE_IDS = np.array([p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+self.URDF,
                                              self.INIT_XYZS[0,:],
                                              p.getQuaternionFromEuler(self.INIT_RPYS[0,:]),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT)])
        
        if self.GUI and self.USER_DEBUG:
            self._showDroneLocalAxes(0)
            
        if self.OBSTACLES:
            self._addObstacles()
    
    ################################################################################

    def _updateAndStoreKinematicInformation(self):
        """Get and store the drone's kinematic information"""
        self.pos[0], self.quat[0] = p.getBasePositionAndOrientation(self.DRONE_IDS[0], physicsClientId=self.CLIENT)
        self.rpy[0] = p.getEulerFromQuaternion(self.quat[0])
        self.vel[0], self.ang_v[0] = p.getBaseVelocity(self.DRONE_IDS[0], physicsClientId=self.CLIENT)
    
    ################################################################################

    def _startVideoRecording(self):
        """Start video recording"""
        if self.RECORD and self.GUI:
            self.VIDEO_ID = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
                                                fileName=os.path.dirname(os.path.abspath(__file__))+"/../../files/videos/video-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".mp4",
                                                physicsClientId=self.CLIENT)
        if self.RECORD and not self.GUI:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.dirname(os.path.abspath(__file__))+"/../../files/videos/video-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+"/"
            os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)
    
    ################################################################################

    def _getDroneStateVector(self, nth_drone=0):
        """Get the complete drone state vector"""
        state = np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
                           self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :]])
        return state.reshape(20,)

    ################################################################################

    def _physics(self, rpm, nth_drone=0):
        """Basic physics implementation for CF2X drone"""
        forces = np.array(rpm**2)*self.KF
        torques = np.array(rpm**2)*self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT)
                                 
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.CLIENT)

    ################################################################################

    def _groundEffect(self, rpm, nth_drone=0):
        """Ground effect implementation"""
        link_states = np.array(p.getLinkStates(self.DRONE_IDS[nth_drone],
                                               linkIndices=[0, 1, 2, 3, 4],
                                               computeLinkVelocity=1,
                                               computeForwardKinematics=1,
                                               physicsClientId=self.CLIENT))
        prop_heights = np.array([link_states[0, 0][2], link_states[1, 0][2], link_states[2, 0][2], link_states[3, 0][2]])
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
        gnd_effects = np.array(rpm**2) * self.KF * self.GND_EFF_COEFF * (self.PROP_RADIUS/(4 * prop_heights))**2
        
        if np.abs(self.rpy[nth_drone,0]) < np.pi/2 and np.abs(self.rpy[nth_drone,1]) < np.pi/2:
            for i in range(4):
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     i,
                                     forceObj=[0, 0, gnd_effects[i]],
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT)

    ################################################################################

    def _drag(self, rpm, nth_drone=0):
        """Drag force implementation"""
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2*np.pi*rpm/60))
        drag = np.dot(base_rot, drag_factors*np.array(self.vel[nth_drone, :]))
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                             4,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.CLIENT)
    
    ################################################################################

    def _downwash(self, nth_drone=0):
        """Downwash effect implementation - removed since only one drone exists"""
        pass

    ################################################################################

    def _dynamics(self, rpm, nth_drone=0):
        """Explicit dynamics implementation for CF2X drone"""
        pos = self.pos[nth_drone,:]
        quat = self.quat[nth_drone,:]
        rpy = self.rpy[nth_drone,:]
        vel = self.vel[nth_drone,:]
        rpy_rates = self.rpy_rates[nth_drone,:]
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        forces = np.array(rpm**2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
        z_torques = np.array(rpm**2)*self.KM
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L/np.sqrt(2))
        y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L/np.sqrt(2))
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
        rpy_rates_deriv = np.dot(self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.M

        vel = vel + self.TIMESTEP * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self.TIMESTEP * rpy_rates_deriv
        pos = pos + self.TIMESTEP * vel
        rpy = rpy + self.TIMESTEP * rpy_rates

        p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                          pos,
                                          p.getQuaternionFromEuler(rpy),
                                          physicsClientId=self.CLIENT)
        p.resetBaseVelocity(self.DRONE_IDS[nth_drone],
                            vel,
                            [-1, -1, -1], 
                            physicsClientId=self.CLIENT)
        self.rpy_rates[nth_drone,:] = rpy_rates
    
    ################################################################################

    def _normalizedActionToRPM(self, action):
        """Convert normalized action to RPM values"""
        if np.any(np.abs(action)) > 1:
            print("\n[ERROR] it", self.step_counter, "in BaseAviary._normalizedActionToRPM(), out-of-bound action")
        return np.where(action <= 0, (action+1)*self.HOVER_RPM, action*self.MAX_RPM)
    
    ################################################################################

    def _saveLastAction(self, action):
        """Store the last action applied"""
        res_action = np.resize(action, (1, 4))
        self.last_action = np.reshape(res_action, (1, 4))
    
    ################################################################################

    def _showDroneLocalAxes(self, nth_drone=0):
        """Display the drone's local axes"""
        if self.GUI:
            AXIS_LENGTH = 2*self.L
            self.X_AX = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                          lineToXYZ=[AXIS_LENGTH, 0, 0],
                                          lineColorRGB=[1, 0, 0],
                                          parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                          parentLinkIndex=-1,
                                          replaceItemUniqueId=int(self.X_AX),
                                          physicsClientId=self.CLIENT)
            self.Y_AX = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                          lineToXYZ=[0, AXIS_LENGTH, 0],
                                          lineColorRGB=[0, 1, 0],
                                          parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                          parentLinkIndex=-1,
                                          replaceItemUniqueId=int(self.Y_AX),
                                          physicsClientId=self.CLIENT)
            self.Z_AX = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                          lineToXYZ=[0, 0, AXIS_LENGTH],
                                          lineColorRGB=[0, 0, 1],
                                          parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                          parentLinkIndex=-1,
                                          replaceItemUniqueId=int(self.Z_AX),
                                          physicsClientId=self.CLIENT)
    
    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment"""
        p.loadURDF("samurai.urdf", physicsClientId=self.CLIENT)
        p.loadURDF("duck_vhacd.urdf",
                   [-.5, -.5, .05],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT)
        p.loadURDF("cube_no_rotation.urdf",
                   [-.5, -2.5, .5],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT)
        p.loadURDF("sphere2.urdf",
                   [0, 2, .5],
                   p.getQuaternionFromEuler([0,0,0]),
                   physicsClientId=self.CLIENT)
    
    ################################################################################
    
    def _parseURDFParameters(self):
        """Parse parameters from the drone's URDF file"""
        URDF_TREE = etxml.parse(os.path.dirname(os.path.abspath(__file__))+"/../assets/cf2x.urdf").getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
               GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3

    ################################################################################
    # The following methods must be implemented by subclasses
    ################################################################################
    
    def _actionSpace(self):
        """Returns the action space of the environment"""
        raise NotImplementedError
    
    ################################################################################
    
    def _observationSpace(self):
        """Returns the observation space of the environment"""
        raise NotImplementedError
    
    ################################################################################
    
    def _computeObs(self):
        """Returns the current observation of the environment"""
        raise NotImplementedError
    
    ################################################################################
    
    def _preprocessAction(self, action):
        """Pre-processes the action passed to step()"""
        raise NotImplementedError
    
    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value"""
        raise NotImplementedError
    
    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value"""
        raise NotImplementedError
    
    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dictionary"""
        raise NotImplementedError
