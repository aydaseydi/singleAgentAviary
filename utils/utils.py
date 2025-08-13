import time
import numpy as np
from scipy.optimize import nnls

def sync(i, start_time, timestep):
    """Sync simulation steps with wall-clock time.
    
    Args:
        i: Current simulation iteration
        start_time: Timestamp when simulation started
        timestep: Desired time between steps (seconds)
    """
    if timestep > 0.04 or i % max(1, int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        remaining = timestep*i - elapsed
        if remaining > 0:
            time.sleep(remaining)

################################################################################

def nnlsRPM(thrust, x_torque, y_torque, z_torque, 
           max_thrust, max_xy_torque, max_z_torque,
           mixer_matrix, inv_mixer_matrix, b_coeff):
    """Compute RPMs using Non-Negative Least Squares for CF2X drone.
    
    Args:
        thrust: Desired thrust (N)
        x_torque: Desired x-axis torque (Nm)
        y_torque: Desired y-axis torque (Nm) 
        z_torque: Desired z-axis torque (Nm)
        max_thrust: Maximum possible thrust (N)
        max_xy_torque: Maximum xy torque (Nm)
        max_z_torque: Maximum z torque (Nm)
        mixer_matrix: CF2X mixer matrix (4x3)
        inv_mixer_matrix: Pseudo-inverse of mixer matrix
        b_coeff: Scaling coefficients
        
    Returns:
        ndarray: (4,) array of RPMs for each motor
    """
    # Scale thrust and torques
    B = np.array([thrust, x_torque, y_torque, z_torque]) * b_coeff
    
    # Calculate squared RPMs
    sq_rpm = inv_mixer_matrix @ B
    
    # Use NNLS if any RPM would be negative
    if np.min(sq_rpm) < 0:
        sq_rpm, _ = nnls(mixer_matrix, B, maxiter=12)
    
    return np.sqrt(np.clip(sq_rpm, 0, None))