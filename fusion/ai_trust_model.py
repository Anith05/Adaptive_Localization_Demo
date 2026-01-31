"""
AI-based sensor trust model.
Estimates reliability of each sensor using rule-based heuristics.
"""
import numpy as np
from collections import deque


class AITrustModel:
    """
    Lightweight AI trust estimator for sensor fusion.
    Uses rule-based heuristics to estimate sensor reliability.
    Output: trust score ∈ [0, 1] per sensor.
    """
    
    def __init__(self, window_size=20):
        """Initialize trust model."""
        self.window_size = window_size
        
        # History buffers for residual tracking
        self.odom_residual_history = deque(maxlen=window_size)
        self.imu_residual_history = deque(maxlen=window_size)
        self.lidar_residual_history = deque(maxlen=window_size)
        
        # Trust scores
        self.odom_trust = 1.0
        self.imu_trust = 1.0
        self.lidar_trust = 1.0
        
        # Thresholds for anomaly detection
        self.odom_threshold = 0.15
        self.imu_threshold = 0.20
        self.lidar_threshold = 0.40
        
    def update_odometry_trust(self, predicted_state, dx_odom, dy_odom, dtheta_odom):
        """
        Update trust for odometry based on consistency with prediction.
        
        Args:
            predicted_state: Current EKF predicted state [x, y, theta]
            dx_odom: Odometry delta x
            dy_odom: Odometry delta y
            dtheta_odom: Odometry delta theta
        """
        # Calculate expected motion magnitude
        motion_magnitude = np.sqrt(dx_odom**2 + dy_odom**2)
        
        # Calculate residual (difference from expected)
        # For odometry, we check if motion is consistent with control
        residual = np.abs(dtheta_odom) + motion_magnitude
        
        # Store residual
        self.odom_residual_history.append(residual)
        
        # Calculate trust based on residual statistics
        if len(self.odom_residual_history) >= 5:
            recent_residuals = list(self.odom_residual_history)[-10:]
            mean_residual = np.mean(recent_residuals)
            std_residual = np.std(recent_residuals)
            
            # Detect anomalies (sudden spikes indicate slip)
            if mean_residual > self.odom_threshold:
                # High residual = low trust (likely wheel slip)
                self.odom_trust = max(0.15, 1.0 - (mean_residual / (self.odom_threshold * 2)))
            else:
                # Gradually restore trust
                self.odom_trust = min(1.0, self.odom_trust + 0.03)
                
            # Penalize high variance (inconsistency)
            if std_residual > 0.08:
                self.odom_trust *= 0.85
        
        return self.odom_trust
    
    def update_imu_trust(self, predicted_theta, measured_theta, angular_velocity):
        """
        Update trust for IMU based on measurement consistency.
        
        Args:
            predicted_theta: Predicted orientation
            measured_theta: Measured orientation from IMU
            angular_velocity: Current angular velocity
        """
        # Calculate innovation (difference between measurement and prediction)
        innovation = measured_theta - predicted_theta
        innovation = np.arctan2(np.sin(innovation), np.cos(innovation))
        
        residual = np.abs(innovation)
        
        # Store residual
        self.imu_residual_history.append(residual)
        
        # Calculate trust
        if len(self.imu_residual_history) >= 5:
            recent_residuals = list(self.imu_residual_history)[-10:]
            mean_residual = np.mean(recent_residuals)
            
            # High angular velocity increases expected noise
            noise_multiplier = 1.0 + np.abs(angular_velocity) * 2.0
            adaptive_threshold = self.imu_threshold * noise_multiplier
            
            if mean_residual > adaptive_threshold:
                # High residual during turns = low trust
                self.imu_trust = max(0.3, 1.0 - (mean_residual / adaptive_threshold))
            else:
                # Restore trust
                self.imu_trust = min(1.0, self.imu_trust + 0.05)
        
        return self.imu_trust
    
    def update_lidar_trust(self, predicted_state, measured_x, measured_y, min_obstacle_dist):
        """
        Update trust for LiDAR based on occlusion and consistency.
        
        Args:
            predicted_state: Current predicted state [x, y, theta]
            measured_x: Measured x from LiDAR
            measured_y: Measured y from LiDAR
            min_obstacle_dist: Minimum distance to nearest obstacle
        """
        # Calculate position innovation
        innovation_x = measured_x - predicted_state[0]
        innovation_y = measured_y - predicted_state[1]
        innovation_magnitude = np.sqrt(innovation_x**2 + innovation_y**2)
        
        # Store residual
        self.lidar_residual_history.append(innovation_magnitude)
        
        # Calculate trust
        if len(self.lidar_residual_history) >= 5:
            recent_residuals = list(self.lidar_residual_history)[-10:]
            mean_residual = np.mean(recent_residuals)
            
            # Occlusion detection: if robot is very close to obstacle, LiDAR is less reliable
            occlusion_penalty = 1.0
            if min_obstacle_dist < 1.8:
                # Smooth penalty based on distance
                occlusion_penalty = max(0.1, min_obstacle_dist / 1.8)
            
            if mean_residual > self.lidar_threshold:
                # High innovation = low trust
                base_trust = max(0.1, 1.0 - (mean_residual / self.lidar_threshold))
                self.lidar_trust = base_trust * occlusion_penalty
            else:
                # Restore trust gradually
                self.lidar_trust = min(1.0, self.lidar_trust + 0.03)
                self.lidar_trust *= occlusion_penalty
        
        return self.lidar_trust
    
    def get_trust_scores(self):
        """Get current trust scores for all sensors."""
        return {
            'odometry': float(self.odom_trust),
            'imu': float(self.imu_trust),
            'lidar': float(self.lidar_trust)
        }
    
    def reset(self):
        """Reset trust model."""
        self.odom_residual_history.clear()
        self.imu_residual_history.clear()
        self.lidar_residual_history.clear()
        self.odom_trust = 1.0
        self.imu_trust = 1.0
        self.lidar_trust = 1.0
