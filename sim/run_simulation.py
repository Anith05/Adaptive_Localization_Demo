"""
Main simulation runner.
TWO-ROBOT COMPARATIVE DEMO: Fixed EKF vs AI-Adaptive EKF
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.pybullet_env import PyBulletEnvironment
from sim.robot import DifferentialDriveRobot
from fusion.ekf import ExtendedKalmanFilter
from fusion.adaptive_fusion import AdaptiveEKF
from demo.live_visualization import LiveVisualization
from demo.pause_controls import PauseControls


class SimulationRunner:
    """Dual-robot comparative simulation."""
    
    def __init__(self, duration=90.0, goal_position=[3.5, 3.5]):
        """Initialize two-robot simulation."""
        self.duration = duration
        self.dt = 0.05  # 20 Hz
        self.goal_position = np.array(goal_position)
        self.goal_threshold = 0.4  # meters
        
        # Goal reached flags
        self.robot_a_goal_reached = False
        self.robot_b_goal_reached = False
        
        # Environment
        self.env = PyBulletEnvironment(gui=True)
        
        # Robot A (RED - Fixed EKF)
        self.robot_a = None
        self.ekf_a = ExtendedKalmanFilter()
        
        # Robot B (GREEN - AI-Adaptive EKF)
        self.robot_b = None
        self.ekf_b = AdaptiveEKF()
        
        # Visualization
        self.viz = LiveVisualization()
        self.pause_ctrl = PauseControls()
        
        # Data logging
        self.log_time = []
        self.log_gt_a = []
        self.log_gt_b = []
        self.log_ekf_a = []
        self.log_ekf_b = []
        self.log_trust_scores = []
        
    def setup(self):
        """Set up two-robot simulation environment."""
        print("Setting up dual-robot environment...")
        
        # Initialize environment first
        self.env.setup()
        
        # Create two robots with different colors and positions
        self.robot_a = DifferentialDriveRobot(self.env, start_pos=[-0.3, -4, 0.1], color=[1, 0, 0, 1])  # RED
        self.robot_b = DifferentialDriveRobot(self.env, start_pos=[0.3, -4, 0.1], color=[0, 1, 0, 1])  # GREEN
        
        self.robot_a.create()
        self.robot_b.create()
        
        # Visualize goal
        self.env.visualize_goal(self.goal_position.tolist())
        
        # Initialize EKF states
        start_a = self.robot_a.get_ground_truth()
        start_b = self.robot_b.get_ground_truth()
        self.ekf_a.reset(start_a)
        self.ekf_b.reset(start_b)
        
        print("Environment ready. Both robots initialized.")
        print(f"Robot A (RED - Fixed EKF) start: {start_a[:2]}")
        print(f"Robot B (GREEN - AI-Adaptive) start: {start_b[:2]}")
        print(f"Goal position: {self.goal_position}")
        print("\nPress SPACE to pause/resume. Close window to quit.")
    
    def compute_goal_directed_control(self, current_pos, current_theta, robot, goal_reached):
        """
        Compute velocity commands towards goal with obstacle avoidance.
        
        Args:
            current_pos: Current [x, y] position
            current_theta: Current heading
            robot: Robot instance for LiDAR
            goal_reached: Whether goal is already reached
        """
        # Get LiDAR scan for obstacle avoidance
        lidar_ranges, _ = robot.get_lidar_scan()
        
        # Goal direction in global frame
        dx_goal = self.goal_position[0] - current_pos[0]
        dy_goal = self.goal_position[1] - current_pos[1]
        distance_to_goal = np.sqrt(dx_goal**2 + dy_goal**2)
        
        # Check if goal reached
        if distance_to_goal < self.goal_threshold or goal_reached:
            return 0.0, 0.0
        
        # Desired heading to goal
        desired_theta = np.arctan2(dy_goal, dx_goal)
        
        # Angular error (normalize to [-pi, pi])
        angle_error = desired_theta - current_theta
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
        # Obstacle detection (check front 90 degrees)
        n_rays = len(lidar_ranges)
        front_indices = list(range(n_rays//4, 3*n_rays//4))
        front_ranges = [lidar_ranges[i] for i in front_indices]
        min_distance = min(front_ranges)
        
        # Control parameters
        max_linear_vel = 0.8  # m/s
        max_angular_vel = 1.5  # rad/s
        obstacle_threshold = 0.6  # meters
        
        # Proportional control
        if min_distance < obstacle_threshold:
            # Obstacle avoidance: slow down and turn away
            v = 0.1
            w = max_angular_vel * np.sign(angle_error)
        else:
            # Normal goal-directed motion
            v = max_linear_vel * min(1.0, distance_to_goal / 2.0)
            w = 2.0 * angle_error  # Proportional angular control
            w = np.clip(w, -max_angular_vel, max_angular_vel)
        
        return v, w
    
    def run(self):
        """Run the dual-robot simulation."""
        t = 0.0
        step_count = 0
        
        while t < self.duration and not (self.robot_a_goal_reached and self.robot_b_goal_reached):
            # Check pause state
            paused = self.pause_ctrl.update()
            
            if not paused:
                # ========== ROBOT A (Fixed EKF) ==========
                # Get ground truth
                gt_a_x, gt_a_y, gt_a_theta = self.robot_a.get_ground_truth()
                
                # Get sensor measurements
                dx_odom_a, dy_odom_a, dtheta_odom_a = self.robot_a.get_wheel_odometry()
                theta_imu_a = self.robot_a.get_imu_orientation()
                x_lidar_a, y_lidar_a = self.robot_a.get_lidar_position_estimate()
                
                # Update Fixed EKF
                self.ekf_a.predict(dx_odom_a, dy_odom_a, dtheta_odom_a)
                self.ekf_a.update_imu(theta_imu_a)
                self.ekf_a.update_lidar(x_lidar_a, y_lidar_a)
                ekf_a_state = self.ekf_a.get_state()
                
                # Control based on ESTIMATED pose (not ground truth)
                v_a, w_a = self.compute_goal_directed_control(
                    ekf_a_state[:2], ekf_a_state[2], 
                    self.robot_a, self.robot_a_goal_reached
                )
                self.robot_a.set_velocity_command(v_a, w_a)
                self.robot_a.update(self.dt)
                
                # Check if Robot A reached goal
                dist_a_to_goal = np.sqrt((gt_a_x - self.goal_position[0])**2 + 
                                        (gt_a_y - self.goal_position[1])**2)
                if dist_a_to_goal < self.goal_threshold and not self.robot_a_goal_reached:
                    self.robot_a_goal_reached = True
                    print(f"\n🎯 ROBOT A (Fixed EKF) reached goal at t={t:.1f}s")
                
                # ========== ROBOT B (AI-Adaptive EKF) ==========
                # Get ground truth
                gt_b_x, gt_b_y, gt_b_theta = self.robot_b.get_ground_truth()
                
                # Get sensor measurements
                dx_odom_b, dy_odom_b, dtheta_odom_b = self.robot_b.get_wheel_odometry()
                theta_imu_b = self.robot_b.get_imu_orientation()
                x_lidar_b, y_lidar_b = self.robot_b.get_lidar_position_estimate()
                distances_b, _ = self.robot_b.get_lidar_scan()
                min_obstacle_dist_b = np.min(distances_b)
                
                # Get angular velocity for trust model (use previous if available)
                w_b_for_trust = w_b if 'w_b' in locals() else 0.0
                
                # Update AI-Adaptive EKF
                self.ekf_b.predict(dx_odom_b, dy_odom_b, dtheta_odom_b)
                self.ekf_b.update_imu(theta_imu_b, w_b_for_trust)
                self.ekf_b.update_lidar(x_lidar_b, y_lidar_b, min_obstacle_dist_b)
                ekf_b_state = self.ekf_b.get_state()
                trust_scores = self.ekf_b.get_trust_scores()
                
                # Control based on ESTIMATED pose (not ground truth)
                v_b, w_b = self.compute_goal_directed_control(
                    ekf_b_state[:2], ekf_b_state[2], 
                    self.robot_b, self.robot_b_goal_reached
                )
                self.robot_b.set_velocity_command(v_b, w_b)
                self.robot_b.update(self.dt)
                
                # Check if Robot B reached goal
                dist_b_to_goal = np.sqrt((gt_b_x - self.goal_position[0])**2 + 
                                        (gt_b_y - self.goal_position[1])**2)
                if dist_b_to_goal < self.goal_threshold and not self.robot_b_goal_reached:
                    self.robot_b_goal_reached = True
                    print(f"\n🎯 ROBOT B (AI-Adaptive) reached goal at t={t:.1f}s")
                
                # Log data
                self.log_time.append(t)
                self.log_gt_a.append([gt_a_x, gt_a_y, gt_a_theta])
                self.log_gt_b.append([gt_b_x, gt_b_y, gt_b_theta])
                self.log_ekf_a.append(ekf_a_state.copy())
                self.log_ekf_b.append(ekf_b_state.copy())
                # Convert trust scores dict to array [odom, imu, lidar]
                self.log_trust_scores.append([trust_scores['odometry'], trust_scores['imu'], trust_scores['lidar']])
                
                # Update visualization
                self.viz.update_dual_robot(
                    [gt_a_x, gt_a_y, gt_a_theta],
                    [gt_b_x, gt_b_y, gt_b_theta],
                    ekf_a_state,
                    ekf_b_state
                )
                
                # Step simulation
                self.env.step()
                
                # Increment time
                t += self.dt
                step_count += 1
                
                # Print progress every 5 seconds
                if int(t) % 5 == 0 and t > 0 and step_count % int(5/self.dt) == 0:
                    status_a = "✓ GOAL" if self.robot_a_goal_reached else f"{dist_a_to_goal:.2f}m"
                    status_b = "✓ GOAL" if self.robot_b_goal_reached else f"{dist_b_to_goal:.2f}m"
                    
                    # Highlight low trust values
                    odom_str = f"O={trust_scores['odometry']:.2f}"
                    if trust_scores['odometry'] < 0.5:
                        odom_str += "⚠️ "
                    
                    imu_str = f"I={trust_scores['imu']:.2f}"
                    if trust_scores['imu'] < 0.5:
                        imu_str += "⚠️ "
                    
                    lidar_str = f"L={trust_scores['lidar']:.2f}"
                    if trust_scores['lidar'] < 0.3:
                        lidar_str += "⚠️ "
                    
                    print(f"t={t:.1f}s | Robot A: {status_a} | Robot B: {status_b} | "
                          f"Trust: {odom_str} {imu_str} {lidar_str}")
            
            # Small delay for real-time visualization
            time.sleep(0.01)
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        if self.robot_a_goal_reached:
            print("✓ Robot A (Fixed EKF): REACHED GOAL")
        else:
            print(f"✗ Robot A (Fixed EKF): {dist_a_to_goal:.2f}m from goal")
            
        if self.robot_b_goal_reached:
            print("✓ Robot B (AI-Adaptive): REACHED GOAL")
        else:
            print(f"✗ Robot B (AI-Adaptive): {dist_b_to_goal:.2f}m from goal")
        print("="*60)
        
        # Generate plots after simulation
        self.generate_plots()
    
    def generate_plots(self):
        """Generate comparison plots for dual robots."""
        print("\nGenerating comparison plots...")
        
        # Convert lists to arrays
        time_array = np.array(self.log_time)
        gt_a_array = np.array(self.log_gt_a)
        gt_b_array = np.array(self.log_gt_b)
        ekf_a_array = np.array(self.log_ekf_a)
        ekf_b_array = np.array(self.log_ekf_b)
        
        # Calculate errors (EKF vs Ground Truth)
        error_a_pos = np.sqrt((ekf_a_array[:, 0] - gt_a_array[:, 0])**2 + 
                              (ekf_a_array[:, 1] - gt_a_array[:, 1])**2)
        error_b_pos = np.sqrt((ekf_b_array[:, 0] - gt_b_array[:, 0])**2 + 
                              (ekf_b_array[:, 1] - gt_b_array[:, 1])**2)
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Trajectory comparison
        ax1 = axes[0, 0]
        ax1.plot(gt_a_array[:, 0], gt_a_array[:, 1], 'r--', linewidth=1.5, alpha=0.5, label='GT Robot A')
        ax1.plot(gt_b_array[:, 0], gt_b_array[:, 1], 'g--', linewidth=1.5, alpha=0.5, label='GT Robot B')
        ax1.plot(ekf_a_array[:, 0], ekf_a_array[:, 1], 'r-', linewidth=2, label='Robot A (Fixed EKF)')
        ax1.plot(ekf_b_array[:, 0], ekf_b_array[:, 1], 'g-', linewidth=2, label='Robot B (AI-Adaptive)')
        ax1.plot(self.goal_position[0], self.goal_position[1], 'yo', markersize=15, label='Goal')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Dual Robot Trajectory Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot 2: Localization error over time
        ax2 = axes[0, 1]
        ax2.plot(time_array, error_a_pos, 'r-', linewidth=1.5, label='Robot A (Fixed EKF)')
        ax2.plot(time_array, error_b_pos, 'g-', linewidth=1.5, label='Robot B (AI-Adaptive)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Localization Error (m)')
        ax2.set_title('Localization Error Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trust scores over time
        ax3 = axes[1, 0]
        trust_array = np.array(self.log_trust_scores)
        if len(trust_array) > 0 and len(trust_array.shape) > 1:
            ax3.plot(time_array, trust_array[:, 0], 'b-', linewidth=1.5, label='Odometry')
            ax3.plot(time_array, trust_array[:, 1], 'orange', linewidth=1.5, label='IMU')
            ax3.plot(time_array, trust_array[:, 2], 'purple', linewidth=1.5, label='LiDAR')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Trust Score')
        ax3.set_title('AI Trust Scores (Robot B)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        # Plot 4: Error statistics
        ax4 = axes[1, 1]
        stats_data = [
            [np.mean(error_a_pos), np.std(error_a_pos), np.max(error_a_pos)],
            [np.mean(error_b_pos), np.std(error_b_pos), np.max(error_b_pos)]
        ]
        x_labels = ['Robot A\n(Fixed)', 'Robot B\n(Adaptive)']
        x_pos = np.arange(len(x_labels))
        width = 0.25
        
        ax4.bar(x_pos - width, [s[0] for s in stats_data], width, label='Mean', color='skyblue')
        ax4.bar(x_pos, [s[1] for s in stats_data], width, label='Std Dev', color='lightcoral')
        ax4.bar(x_pos + width, [s[2] for s in stats_data], width, label='Max', color='lightgreen')
        
        ax4.set_ylabel('Error (m)')
        ax4.set_title('Localization Error Statistics')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(x_labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, 'dual_robot_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plots saved to: {save_path}")
        
        plt.show()


def main():
    """Run the dual-robot comparative demonstration."""
    print("="*60)
    print("TWO-ROBOT COMPARATIVE LOCALIZATION DEMO")
    print("="*60)
    print("Robot A (RED): Fixed-weight EKF localization")
    print("Robot B (GREEN): AI-assisted adaptive sensor fusion")
    print("\nBoth robots navigate from start to goal under identical")
    print("physical disturbances (wheel slip, IMU noise, occlusion).")
    print("\nObserve how Robot B (AI-Adaptive) maintains better")
    print("localization accuracy by dynamically adjusting sensor trust.")
    print("="*60 + "\n")
    
    # Define goal position
    goal = [3.5, 3.5]
    
    # Run simulation (90 seconds for clear convergence)
    sim = SimulationRunner(duration=90.0, goal_position=goal)
    sim.setup()
    sim.run()


if __name__ == "__main__":
    main()
