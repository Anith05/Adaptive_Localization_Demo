"""
Adaptive EKF that uses AI trust scores to adjust sensor fusion.
"""
import numpy as np
from fusion.ekf import ExtendedKalmanFilter
from fusion.ai_trust_model import AITrustModel


class AdaptiveEKF:
    """
    Adaptive Extended Kalman Filter with AI-based sensor trust.
    Scales measurement covariance based on trust scores.
    """
    
    def __init__(self):
        """Initialize adaptive EKF."""
        # Base EKF
        self.ekf = ExtendedKalmanFilter()
        
        # AI trust model
        self.trust_model = AITrustModel()
        
        # Store base measurement covariances
        self.R_odom_base = self.ekf.R_odom.copy()
        self.R_imu_base = self.ekf.R_imu.copy()
        self.R_lidar_base = self.ekf.R_lidar.copy()
        
    def predict(self, dx_odom, dy_odom, dtheta_odom):
        """
        Prediction step using odometry.
        Updates odometry trust based on motion consistency.
        """
        # Get current state before prediction
        current_state = self.ekf.get_state()
        
        # Update odometry trust
        self.trust_model.update_odometry_trust(
            current_state, dx_odom, dy_odom, dtheta_odom
        )
        
        # Perform EKF prediction
        self.ekf.predict(dx_odom, dy_odom, dtheta_odom)
        
    def update_imu(self, theta_measured, angular_velocity):
        """
        Update with IMU measurement using adaptive trust.
        
        Args:
            theta_measured: Measured orientation
            angular_velocity: Current angular velocity (for trust calculation)
        """
        # Get current predicted state
        predicted_state = self.ekf.get_state()
        
        # Update IMU trust
        trust = self.trust_model.update_imu_trust(
            predicted_state[2], theta_measured, angular_velocity
        )
        
        # Scale measurement covariance inversely with trust
        # Low trust = high covariance = low weight in fusion
        trust_factor = max(0.01, trust)  # Avoid division by zero
        self.ekf.R_imu = self.R_imu_base / trust_factor
        
        # Perform EKF update
        self.ekf.update_imu(theta_measured)
        
    def update_lidar(self, x_measured, y_measured, min_obstacle_dist):
        """
        Update with LiDAR measurement using adaptive trust.
        
        Args:
            x_measured: Measured x position
            y_measured: Measured y position
            min_obstacle_dist: Minimum distance to nearest obstacle
        """
        # Get current predicted state
        predicted_state = self.ekf.get_state()
        
        # Update LiDAR trust
        trust = self.trust_model.update_lidar_trust(
            predicted_state, x_measured, y_measured, min_obstacle_dist
        )
        
        # Scale measurement covariance inversely with trust
        trust_factor = max(0.01, trust)
        self.ekf.R_lidar = self.R_lidar_base / trust_factor
        
        # Perform EKF update
        self.ekf.update_lidar(x_measured, y_measured)
        
    def get_state(self):
        """Get current state estimate."""
        return self.ekf.get_state()
    
    def get_covariance(self):
        """Get current covariance."""
        return self.ekf.get_covariance()
    
    def get_trust_scores(self):
        """Get current trust scores."""
        return self.trust_model.get_trust_scores()
    
    def reset(self, initial_state=None):
        """Reset adaptive EKF."""
        self.ekf.reset(initial_state)
        self.trust_model.reset()
        
        # Restore base covariances
        self.ekf.R_odom = self.R_odom_base.copy()
        self.ekf.R_imu = self.R_imu_base.copy()
        self.ekf.R_lidar = self.R_lidar_base.copy()
