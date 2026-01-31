"""
Classical Extended Kalman Filter for robot localization.
Fixed sensor noise and fusion weights.
"""
import numpy as np


class ExtendedKalmanFilter:
    """
    Classical EKF for 2D robot localization.
    State: [x, y, theta]
    """
    
    def __init__(self):
        """Initialize EKF with fixed parameters."""
        # State: [x, y, theta]
        self.state = np.array([0.0, -4.0, 0.0])
        
        # State covariance
        self.P = np.eye(3) * 0.1
        
        # Process noise covariance (fixed)
        self.Q = np.diag([0.01, 0.01, 0.005])
        
        # Measurement noise covariances (fixed)
        self.R_odom = np.diag([0.04, 0.04, 0.04])      # Odometry
        self.R_imu = np.array([[0.01]])                 # IMU (orientation only)
        self.R_lidar = np.diag([0.25, 0.25])           # LiDAR (position)
        
    def predict(self, dx_odom, dy_odom, dtheta_odom):
        """
        Prediction step using odometry motion model.
        
        Args:
            dx_odom: Change in x from odometry
            dy_odom: Change in y from odometry
            dtheta_odom: Change in theta from odometry
        """
        # Current state
        x, y, theta = self.state
        
        # Predict new state
        # Rotate odometry delta to global frame
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        dx_global = dx_odom * cos_theta - dy_odom * sin_theta
        dy_global = dx_odom * sin_theta + dy_odom * cos_theta
        
        # Update state
        self.state[0] += dx_global
        self.state[1] += dy_global
        self.state[2] += dtheta_odom
        
        # Normalize angle
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
        
        # Jacobian of motion model
        F = np.array([
            [1, 0, -dx_odom * sin_theta - dy_odom * cos_theta],
            [0, 1,  dx_odom * cos_theta - dy_odom * sin_theta],
            [0, 0, 1]
        ])
        
        # Update covariance
        self.P = F @ self.P @ F.T + self.Q
        
    def update_imu(self, theta_measured):
        """
        Update step using IMU orientation measurement.
        
        Args:
            theta_measured: Measured orientation from IMU
        """
        # Measurement model: z = theta
        H = np.array([[0, 0, 1]])
        
        # Innovation
        theta_predicted = self.state[2]
        innovation = theta_measured - theta_predicted
        
        # Normalize innovation
        innovation = np.arctan2(np.sin(innovation), np.cos(innovation))
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_imu
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state += K.flatten() * innovation
        
        # Normalize angle
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
        
        # Update covariance
        self.P = (np.eye(3) - K @ H) @ self.P
        
    def update_lidar(self, x_measured, y_measured):
        """
        Update step using LiDAR position measurement.
        
        Args:
            x_measured: Measured x position from LiDAR
            y_measured: Measured y position from LiDAR
        """
        # Measurement model: z = [x, y]
        H = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        
        # Innovation
        z = np.array([x_measured, y_measured])
        z_predicted = self.state[:2]
        innovation = z - z_predicted
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R_lidar
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state += K @ innovation
        
        # Normalize angle
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
        
        # Update covariance
        self.P = (np.eye(3) - K @ H) @ self.P
        
    def get_state(self):
        """Get current state estimate."""
        return self.state.copy()
    
    def get_covariance(self):
        """Get current covariance."""
        return self.P.copy()
    
    def reset(self, initial_state=None):
        """Reset EKF to initial state."""
        if initial_state is None:
            self.state = np.array([0.0, -4.0, 0.0])
        else:
            self.state = np.array(initial_state)
        self.P = np.eye(3) * 0.1
