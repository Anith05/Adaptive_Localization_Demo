"""
Differential drive robot with wheel odometry, IMU, and LiDAR sensors.
"""
import pybullet as p
import numpy as np


class DifferentialDriveRobot:
    """Differential drive robot with physics-based sensor simulation."""
    
    def __init__(self, environment, start_pos=[0, -4, 0.1], color=[0.2, 0.4, 0.8, 1]):
        """Initialize the robot."""
        self.env = environment
        self.start_pos = start_pos
        self.color = color  # RGBA color
        self.robot_id = None
        self.head_id = None  # Dog head
        self.leg_ids = []    # 4 legs
        self.wheel_radius = 0.05  # 5cm wheels
        self.wheel_base = 0.3     # 30cm between wheels
        
        # Control commands
        self.v_cmd = 0.0  # Linear velocity command
        self.w_cmd = 0.0  # Angular velocity command
        
        # Ground truth state
        self.true_x = start_pos[0]
        self.true_y = start_pos[1]
        self.true_theta = 0.0
        
        # Previous odometry for delta calculations
        self.prev_odom_x = start_pos[0]
        self.prev_odom_y = start_pos[1]
        self.prev_odom_theta = 0.0
        
        # Sensor noise parameters
        self.odom_noise_std = 0.02
        self.imu_noise_std = 0.05
        self.lidar_noise_std = 0.03
        
        # IMU noise spike tracking
        self.angular_velocity_history = []
        
    def create(self):
        """Create dog-like robot in PyBullet."""
        # Dog body - elongated box with legs
        body_length = 0.5
        body_width = 0.18
        body_height = 0.08
        leg_radius = 0.03
        leg_height = 0.12
        
        # Create compound collision shape for body
        body_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[body_length/2, body_width/2, body_height/2]
        )
        
        # Visual shape for body
        body_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[body_length/2, body_width/2, body_height/2],
            rgbaColor=self.color
        )
        
        # Create robot body
        self.robot_id = p.createMultiBody(
            baseMass=5.0,
            baseCollisionShapeIndex=body_col,
            baseVisualShapeIndex=body_vis,
            basePosition=self.start_pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, self.true_theta])
        )
        
        # Add head (forward pointing nose)
        head_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=[c * 0.9 for c in self.color[:3]] + [1.0]
        )
        # Create separate visual body for head
        self.head_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=head_vis,
            basePosition=[self.start_pos[0] + body_length/2 + 0.08, self.start_pos[1], self.start_pos[2] + 0.05],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # Add 4 legs as visual cylinders
        leg_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=leg_radius,
            length=leg_height,
            rgbaColor=[c * 0.7 for c in self.color[:3]] + [1.0]
        )
        
        leg_positions = [
            [body_length/4, body_width/2, -leg_height/2],     # Front right
            [body_length/4, -body_width/2, -leg_height/2],    # Front left
            [-body_length/4, body_width/2, -leg_height/2],    # Back right
            [-body_length/4, -body_width/2, -leg_height/2]    # Back left
        ]
        
        self.leg_ids = []
        for i, leg_offset in enumerate(leg_positions):
            leg_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=leg_vis,
                basePosition=[
                    self.start_pos[0] + leg_offset[0],
                    self.start_pos[1] + leg_offset[1],
                    self.start_pos[2] + leg_offset[2]
                ],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
            )
            self.leg_ids.append(leg_id)
        
        # Set initial friction
        p.changeDynamics(self.robot_id, -1, lateralFriction=0.8)
    
    def update(self, dt):
        """
        Update robot physics and move head/legs to follow body.
        
        Args:
            dt: Time step
        """
        # Get robot body pose
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        self.true_x, self.true_y = pos[0], pos[1]
        
        # Get orientation
        euler = p.getEulerFromQuaternion(orn)
        self.true_theta = euler[2]
        
        # Update head position (in front of body)
        head_offset = 0.33  # Distance in front
        head_x = self.true_x + head_offset * np.cos(self.true_theta)
        head_y = self.true_y + head_offset * np.sin(self.true_theta)
        p.resetBasePositionAndOrientation(
            self.head_id,
            [head_x, head_y, pos[2] + 0.05],
            orn
        )
        
        # Update leg positions
        body_length = 0.5
        body_width = 0.18
        leg_height = 0.12
        
        leg_offsets_local = [
            [body_length/4, body_width/2, -leg_height/2],     # Front right
            [body_length/4, -body_width/2, -leg_height/2],    # Front left
            [-body_length/4, body_width/2, -leg_height/2],    # Back right
            [-body_length/4, -body_width/2, -leg_height/2]    # Back left
        ]
        
        for leg_id, leg_offset in zip(self.leg_ids, leg_offsets_local):
            # Rotate offset by robot orientation
            cos_t = np.cos(self.true_theta)
            sin_t = np.sin(self.true_theta)
            leg_x = self.true_x + (leg_offset[0] * cos_t - leg_offset[1] * sin_t)
            leg_y = self.true_y + (leg_offset[0] * sin_t + leg_offset[1] * cos_t)
            p.resetBasePositionAndOrientation(
                leg_id,
                [leg_x, leg_y, pos[2] + leg_offset[2]],
                orn
            )
        
    def set_velocity_command(self, v, w):
        """Set velocity commands for the robot."""
        self.v_cmd = v
        self.w_cmd = w
        
    def update(self, dt):
        """Update robot physics based on velocity commands."""
        # Get current position and orientation
        pos, quat = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(quat)
        
        # Update ground truth
        self.true_x = pos[0]
        self.true_y = pos[1]
        self.true_theta = euler[2]
        
        # Apply friction based on position
        friction = self.env.get_friction_at_position(self.true_x, self.true_y)
        p.changeDynamics(self.robot_id, -1, lateralFriction=friction)
        
        # Convert velocity commands to wheel velocities
        v_right = (2 * self.v_cmd + self.w_cmd * self.wheel_base) / (2 * self.wheel_radius)
        v_left = (2 * self.v_cmd - self.w_cmd * self.wheel_base) / (2 * self.wheel_radius)
        
        # Apply velocity
        vel_x = self.v_cmd * np.cos(self.true_theta)
        vel_y = self.v_cmd * np.sin(self.true_theta)
        
        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=[vel_x, vel_y, 0],
            angularVelocity=[0, 0, self.w_cmd]
        )
        
    def get_ground_truth(self):
        """Get ground truth position and orientation."""
        return self.true_x, self.true_y, self.true_theta
    
    def get_wheel_odometry(self):
        """
        Get wheel odometry with errors from friction/slip.
        Returns delta measurements.
        """
        # Base odometry from ground truth
        dx_true = self.true_x - self.prev_odom_x
        dy_true = self.true_y - self.prev_odom_y
        dtheta_true = self.true_theta - self.prev_odom_theta
        
        # Normalize angle
        dtheta_true = np.arctan2(np.sin(dtheta_true), np.cos(dtheta_true))
        
        # Get current friction (low friction = high slip)
        friction = self.env.get_friction_at_position(self.true_x, self.true_y)
        slip_factor = max(0, 1.0 - friction)  # Higher slip in low friction
        
        # Add slip error (proportional to velocity and slip)
        slip_error = slip_factor * 0.5  # Up to 50% error in low friction zones
        
        # Odometry measurement with slip and noise
        dx_odom = dx_true * (1.0 - slip_error) + np.random.normal(0, self.odom_noise_std) * (1 + slip_factor)
        dy_odom = dy_true * (1.0 - slip_error) + np.random.normal(0, self.odom_noise_std) * (1 + slip_factor)
        dtheta_odom = dtheta_true + np.random.normal(0, self.odom_noise_std * 3) * (1 + slip_factor * 2)
        
        # Update previous odometry
        self.prev_odom_x = self.true_x
        self.prev_odom_y = self.true_y
        self.prev_odom_theta = self.true_theta
        
        return dx_odom, dy_odom, dtheta_odom
    
    def get_imu_orientation(self):
        """
        Get IMU orientation with noise that increases during sharp turns.
        """
        # Track angular velocity
        self.angular_velocity_history.append(abs(self.w_cmd))
        if len(self.angular_velocity_history) > 10:
            self.angular_velocity_history.pop(0)
            
        # Noise increases with angular velocity
        avg_angular_vel = np.mean(self.angular_velocity_history)
        noise_multiplier = 1.0 + avg_angular_vel * 5.0  # More noise during turns
        
        # IMU measurement with increased noise during turns
        theta_imu = self.true_theta + np.random.normal(0, self.imu_noise_std * noise_multiplier)
        
        # Normalize angle
        theta_imu = np.arctan2(np.sin(theta_imu), np.cos(theta_imu))
        
        return theta_imu
    
    def get_lidar_scan(self, num_rays=16):
        """
        Get LiDAR scan using ray casting with occlusion.
        Returns: distances and their corresponding angles.
        """
        angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
        distances = []
        
        max_range = 5.0
        
        for angle in angles:
            # Ray direction in world frame
            ray_angle = self.true_theta + angle
            ray_end = [
                self.true_x + max_range * np.cos(ray_angle),
                self.true_y + max_range * np.sin(ray_angle),
                0.1
            ]
            
            # Perform ray cast
            result = p.rayTest(
                [self.true_x, self.true_y, 0.1],
                ray_end
            )
            
            if result and result[0][0] != -1:
                # Hit something
                hit_fraction = result[0][2]
                distance = max_range * hit_fraction
            else:
                # No hit
                distance = max_range
                
            # Add noise
            distance += np.random.normal(0, self.lidar_noise_std)
            distance = max(0.1, min(distance, max_range))
            
            distances.append(distance)
            
        return np.array(distances), angles
    
    def get_lidar_position_estimate(self):
        """
        Estimate position from LiDAR by detecting nearest wall/obstacle.
        This is a simplified landmark-based localization.
        """
        distances, angles = self.get_lidar_scan(num_rays=16)
        
        # Find minimum distance (nearest obstacle)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        min_angle = angles[min_idx]
        
        # Estimate position relative to nearest landmark
        # This is simplified - in reality would use SLAM or known map
        # For this demo, we'll use the scan to estimate position with occlusion effects
        
        # Occlusion check: if robot is behind obstacle, measurements are less reliable
        # Check if there are nearby obstacles causing occlusion
        obstacle_nearby = min_dist < 1.5
        
        # Use weighted average of ray endpoints to estimate position
        x_est = 0.0
        y_est = 0.0
        weight_sum = 0.0
        
        for dist, angle in zip(distances, angles):
            if dist < 4.5:  # Only use close measurements
                ray_angle = self.true_theta + angle
                # Landmark position
                lx = self.true_x + dist * np.cos(ray_angle)
                ly = self.true_y + dist * np.sin(ray_angle)
                
                weight = 1.0 / (dist + 0.1)  # Closer landmarks have higher weight
                x_est += lx * weight
                y_est += ly * weight
                weight_sum += weight
                
        if weight_sum > 0:
            x_est /= weight_sum
            y_est /= weight_sum
            
            # Back-calculate robot position (simplified)
            # This is a crude approximation
            x_robot = self.true_x + np.random.normal(0, 0.3 * (1 if obstacle_nearby else 0.5))
            y_robot = self.true_y + np.random.normal(0, 0.3 * (1 if obstacle_nearby else 0.5))
        else:
            x_robot = self.true_x
            y_robot = self.true_y
            
        return x_robot, y_robot
    
    def reset(self):
        """Reset robot to initial position."""
        p.resetBasePositionAndOrientation(
            self.robot_id,
            self.start_pos,
            p.getQuaternionFromEuler([0, 0, 0])
        )
        self.true_x = self.start_pos[0]
        self.true_y = self.start_pos[1]
        self.true_theta = 0.0
        self.prev_odom_x = self.start_pos[0]
        self.prev_odom_y = self.start_pos[1]
        self.prev_odom_theta = 0.0
