import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

class Logger:
    """Optimized logging and visualization class for drone simulations."""
    
    # Constants for state indices
    POS_X, POS_Y, POS_Z = 0, 1, 2
    VEL_X, VEL_Y, VEL_Z = 3, 4, 5
    ROLL, PITCH, YAW = 6, 7, 8
    ANG_VEL_X, ANG_VEL_Y, ANG_VEL_Z = 9, 10, 11
    RPM_0, RPM_1, RPM_2, RPM_3 = 12, 13, 14, 15

    def __init__(self, logging_freq_hz: int, num_drones: int = 1, duration_sec: int = 0):
        """
        Initialize logger with specified parameters.
        
        Args:
            logging_freq_hz: Logging frequency in Hz
            num_drones: Number of drones to track
            duration_sec: Optional pre-allocation duration in seconds
        """
        self.LOGGING_FREQ_HZ = logging_freq_hz
        self.NUM_DRONES = num_drones
        
        # Initialize data storage
        self._init_data_arrays(duration_sec)
        self.counters = np.zeros(num_drones, dtype=int)
        
        # Configure plotting style
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + 
                                  cycler('linestyle', ['-', '--', ':', '-.'])))

    def _init_data_arrays(self, duration_sec):
        """Initialize data storage arrays."""
        if duration_sec > 0:
            size = duration_sec * self.LOGGING_FREQ_HZ
            self.timestamps = np.zeros((self.NUM_DRONES, size))
            self.states = np.zeros((self.NUM_DRONES, 16, size))
            self.controls = np.zeros((self.NUM_DRONES, 12, size))
            self.PREALLOCATED = True
        else:
            self.timestamps = np.zeros((self.NUM_DRONES, 0))
            self.states = np.zeros((self.NUM_DRONES, 16, 0))
            self.controls = np.zeros((self.NUM_DRONES, 12, 0))
            self.PREALLOCATED = False

    def log(self, drone: int, timestamp: float, state: np.ndarray, control: np.ndarray = None):
        """
        Log drone state and control information.
        
        Args:
            drone: Drone index (0-based)
            timestamp: Current simulation time
            state: State vector (20 elements)
            control: Control vector (12 elements)
        """
        if control is None:
            control = np.zeros(12)
            
        if not self._validate_input(drone, timestamp, state, control):
            return

        idx = self._get_log_index(drone)
        self.timestamps[drone, idx] = timestamp
        
        # Store reordered state information
        self.states[drone, :, idx] = np.array([
            state[0], state[1], state[2],        # Position (x,y,z)
            state[10], state[11], state[12],     # Velocity (vx,vy,vz)
            state[7], state[8], state[9],        # Orientation (roll,pitch,yaw)
            state[13], state[14], state[15],     # Angular velocity (wx,wy,wz)
            state[16], state[17], state[18], state[19]  # RPMs
        ])
        
        self.controls[drone, :, idx] = control
        self.counters[drone] += 1

    def _validate_input(self, drone, timestamp, state, control):
        """Validate input parameters before logging."""
        if drone < 0 or drone >= self.NUM_DRONES:
            print(f"[ERROR] Invalid drone index: {drone}")
            return False
        if timestamp < 0:
            print(f"[ERROR] Negative timestamp: {timestamp}")
            return False
        if len(state) != 20:
            print(f"[ERROR] State vector length {len(state)} != 20")
            return False
        if len(control) != 12:
            print(f"[ERROR] Control vector length {len(control)} != 12")
            return False
        return True

    def _get_log_index(self, drone):
        """Get current log index and expand arrays if needed."""
        idx = self.counters[drone]
        
        if not self.PREALLOCATED and idx >= self.timestamps.shape[1]:
            self._expand_arrays()
            return self.timestamps.shape[1] - 1
        return idx

    def _expand_arrays(self):
        """Expand storage arrays dynamically."""
        expand_size = max(100, self.LOGGING_FREQ_HZ)  # Expand by at least 100 or 1 second
        empty_chunk = np.zeros((self.NUM_DRONES, expand_size))
        
        self.timestamps = np.concatenate((self.timestamps, empty_chunk), axis=1)
        self.states = np.concatenate((self.states, np.zeros((self.NUM_DRONES, 16, expand_size))), axis=2)
        self.controls = np.concatenate((self.controls, np.zeros((self.NUM_DRONES, 12, expand_size))), axis=2)

    def save(self, filename=None):
        """Save logged data to numpy file."""
        if filename is None:
            filename = f"logs/save-flight-{datetime.now().strftime('%m.%d.%Y_%H.%M.%S')}.npy"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename, timestamps=self.timestamps, states=self.states, controls=self.controls)

    def save_as_csv(self, comment=""):
        """Save data as CSV files organized by drone and metric."""
        csv_dir = f"csv_logs/save-flight-{comment}-{datetime.now().strftime('%m.%d.%Y_%H.%M.%S')}"
        os.makedirs(csv_dir, exist_ok=True)
        
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)
        
        for drone in range(self.NUM_DRONES):
            # Save position data
            self._save_csv(csv_dir, drone, "x", t, self.states[drone, self.POS_X, :])
            self._save_csv(csv_dir, drone, "y", t, self.states[drone, self.POS_Y, :])
            self._save_csv(csv_dir, drone, "z", t, self.states[drone, self.POS_Z, :])
            
            # Save orientation data
            self._save_csv(csv_dir, drone, "roll", t, self.states[drone, self.ROLL, :])
            self._save_csv(csv_dir, drone, "pitch", t, self.states[drone, self.PITCH, :])
            self._save_csv(csv_dir, drone, "yaw", t, self.states[drone, self.YAW, :])
            
            # Save velocity data
            self._save_csv(csv_dir, drone, "vx", t, self.states[drone, self.VEL_X, :])
            self._save_csv(csv_dir, drone, "vy", t, self.states[drone, self.VEL_Y, :])
            self._save_csv(csv_dir, drone, "vz", t, self.states[drone, self.VEL_Z, :])
            
            # Save angular velocity data
            self._save_csv(csv_dir, drone, "wx", t, self.states[drone, self.ANG_VEL_X, :])
            self._save_csv(csv_dir, drone, "wy", t, self.states[drone, self.ANG_VEL_Y, :])
            self._save_csv(csv_dir, drone, "wz", t, self.states[drone, self.ANG_VEL_Z, :])
            
            # Save RPM data
            for i, label in enumerate(['rpm0', 'rpm1', 'rpm2', 'rpm3']):
                self._save_csv(csv_dir, drone, label, t, self.states[drone, self.RPM_0 + i, :])

    def _save_csv(self, dir_path, drone, metric, t, values):
        """Helper method to save single metric to CSV."""
        filename = f"{dir_path}/drone{drone}_{metric}.csv"
        np.savetxt(filename, np.column_stack((t, values)), delimiter=",", header="time,value", comments="")

    def plot(self, pwm=False):
        """Generate comprehensive plots of logged data."""
        fig, axs = plt.subplots(10, 2, figsize=(15, 20))
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)
        
        # Column 1: Position and Orientation
        self._plot_metric(axs[0, 0], t, "x (m)", self.POS_X)
        self._plot_metric(axs[1, 0], t, "y (m)", self.POS_Y)
        self._plot_metric(axs[2, 0], t, "z (m)", self.POS_Z)
        self._plot_metric(axs[3, 0], t, "roll (rad)", self.ROLL)
        self._plot_metric(axs[4, 0], t, "pitch (rad)", self.PITCH)
        self._plot_metric(axs[5, 0], t, "yaw (rad)", self.YAW)
        self._plot_metric(axs[6, 0], t, "wx (rad/s)", self.ANG_VEL_X)
        self._plot_metric(axs[7, 0], t, "wy (rad/s)", self.ANG_VEL_Y)
        self._plot_metric(axs[8, 0], t, "wz (rad/s)", self.ANG_VEL_Z)
        
        # Column 2: Velocities and RPMs/PWMs
        self._plot_metric(axs[0, 1], t, "vx (m/s)", self.VEL_X)
        self._plot_metric(axs[1, 1], t, "vy (m/s)", self.VEL_Y)
        self._plot_metric(axs[2, 1], t, "vz (m/s)", self.VEL_Z)
        
        # Plot RPMs or PWMs
        labels = ['PWM' if pwm else 'RPM'] * 4
        self._plot_metric(axs[6, 1], t, f"{labels[0]}0", self.RPM_0, pwm)
        self._plot_metric(axs[7, 1], t, f"{labels[1]}1", self.RPM_1, pwm)
        self._plot_metric(axs[8, 1], t, f"{labels[2]}2", self.RPM_2, pwm)
        self._plot_metric(axs[9, 1], t, f"{labels[3]}3", self.RPM_3, pwm)

        # Formatting
        for row in axs:
            for ax in row:
                ax.grid(True)
                ax.legend(loc='upper right')
        
        fig.tight_layout()
        plt.show()

    def _plot_metric(self, ax, t, ylabel, state_idx, convert_to_pwm=False):
        """Helper method to plot a single metric."""
        for drone in range(self.NUM_DRONES):
            data = self.states[drone, state_idx, :]
            if convert_to_pwm and drone > 0:  # Skip conversion for drone 0
                data = (data - 4070.3) / 0.2685
            ax.plot(t, data, label=f"drone_{drone}")
        ax.set_xlabel('time (s)')
        ax.set_ylabel(ylabel)