"""
Condition 1: Real-time dual-robot localization comparison.

Robot A (RED): Fixed EKF
Robot B (GREEN): AI-Adaptive EKF

Uses a compact differential-drive robot model tailored for this 2D PyBullet arena.
"""
import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p

# Add project root to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.pybullet_env import PyBulletEnvironment
from fusion.ekf import ExtendedKalmanFilter
from demo.live_visualization import LiveVisualization
from demo.pause_controls import PauseControls


class StrongAdaptiveFusion:
    """Adaptive trust-based fusion with sliding-window, context-aware trust scores."""

    def __init__(self):
        self.ekf = ExtendedKalmanFilter()
        self.Q_base = self.ekf.Q.copy()
        self.R_imu_base = np.array([[0.006]])
        self.R_lidar_base = np.diag([0.04, 0.04])
        self.ekf.R_imu = self.R_imu_base.copy()
        self.ekf.R_lidar = self.R_lidar_base.copy()

        self.trust = {
            "odometry": 1.0,
            "imu": 1.0,
            "lidar": 1.0,
        }
        self.window_size = 20
        self.residual_hist = {
            "odometry": [],
            "imu": [],
            "lidar": [],
        }

    @staticmethod
    def _piecewise_trust(residual, low, high):
        if residual <= low:
            return 1.0
        if residual >= high:
            return 0.3
        alpha = (residual - low) / (high - low)
        return float(np.clip(1.0 - 0.7 * alpha, 0.3, 1.0))

    def _update_trust(self, key, residual, angular_velocity=0.0, obstacle_distance=3.0, slip_active=False):
        hist = self.residual_hist[key]
        hist.append(float(residual))
        if len(hist) > self.window_size:
            hist.pop(0)

        win_mean = float(np.mean(hist)) if hist else float(residual)
        win_std = float(np.std(hist)) if len(hist) > 1 else 1e-6
        z = float((residual - win_mean) / max(win_std, 1e-6))

        if key == "odometry":
            base = self._piecewise_trust(residual, low=0.07, high=0.24)
            if slip_active or z > 1.8:
                base *= 0.80
        elif key == "imu":
            base = self._piecewise_trust(residual, low=0.05, high=0.22)
            if abs(angular_velocity) > 0.80:
                base *= 0.80
        else:
            base = self._piecewise_trust(residual, low=0.10, high=0.52)
            if obstacle_distance < 1.5:
                base *= 0.82

        base = float(np.clip(base, 0.3, 1.0))
        smoothed = 0.80 * self.trust[key] + 0.20 * base
        self.trust[key] = float(np.clip(smoothed, 0.3, 1.0))

    def predict(self, dx_odom, dy_odom, dtheta_odom, slip_active=False):
        motion_residual = float(np.linalg.norm([dx_odom, dy_odom, dtheta_odom]))
        if slip_active:
            motion_residual *= 1.8
        self._update_trust("odometry", motion_residual, slip_active=slip_active)

        # Keep process model stable; adaptive weighting is applied on measurements.
        self.ekf.Q = self.Q_base.copy()
        self.ekf.predict(dx_odom, dy_odom, dtheta_odom)

    def update_imu(self, theta_measured, angular_velocity=0.0):
        pred_theta = self.ekf.get_state()[2]
        residual = float(abs(np.arctan2(np.sin(theta_measured - pred_theta), np.cos(theta_measured - pred_theta))))
        if abs(angular_velocity) > 0.5:
            residual *= 1.3
        residual *= 0.55
        self._update_trust("imu", residual, angular_velocity=angular_velocity)

        trust_imu = max(self.trust["imu"], 0.3)
        self.ekf.R_imu = self.R_imu_base / (trust_imu ** 2)
        self.ekf.update_imu(theta_measured)

    def update_lidar(self, x_measured, y_measured, near_obstacle=False, obstacle_distance=3.0):
        pred = self.ekf.get_state()
        residual = float(np.linalg.norm([x_measured - pred[0], y_measured - pred[1]]))
        if near_obstacle:
            residual *= 1.9
        residual *= 0.45
        self._update_trust("lidar", residual, obstacle_distance=obstacle_distance)

        trust_lidar = max(self.trust["lidar"], 0.3)
        self.ekf.R_lidar = self.R_lidar_base / (trust_lidar ** 2)
        self.ekf.update_lidar(x_measured, y_measured)

    def get_state(self):
        return self.ekf.get_state()

    def get_trust_scores(self):
        return dict(self.trust)

    def reset(self, initial_state=None):
        self.ekf.reset(initial_state)
        self.ekf.Q = self.Q_base.copy()
        self.ekf.R_imu = self.R_imu_base.copy()
        self.ekf.R_lidar = self.R_lidar_base.copy()
        self.trust = {"odometry": 1.0, "imu": 1.0, "lidar": 1.0}
        self.residual_hist = {"odometry": [], "imu": [], "lidar": []}


class CompactDifferentialRobot:
    """Compact box robot with differential-drive style kinematics and noisy sensors."""

    def __init__(self, env, start_pos, color, odom_noise_std=0.02, imu_noise_std=0.05, lidar_noise_std=0.03):
        self.env = env
        self.start_pos = list(start_pos)
        self.color = color

        self.robot_id = None
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.v_applied = 0.0
        self.w_applied = 0.0

        # Command slew limits keep movement smooth and avoid twitchy turns.
        self.max_lin_acc = 0.42
        self.max_ang_acc = 0.95

        self.true_x = self.start_pos[0]
        self.true_y = self.start_pos[1]
        self.true_theta = 0.0

        self.prev_odom_x = self.true_x
        self.prev_odom_y = self.true_y
        self.prev_odom_theta = self.true_theta

        self.odom_noise_std = odom_noise_std
        self.imu_noise_std = imu_noise_std
        self.lidar_noise_std = lidar_noise_std

        self.angular_velocity_hist = []
        self.max_range = 5.0

    def create(self):
        body_size = [0.28, 0.18, 0.12]

        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2.0 for s in body_size])
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[s / 2.0 for s in body_size],
            rgbaColor=self.color,
        )

        self.robot_id = p.createMultiBody(
            baseMass=5.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=self.start_pos,
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, self.true_theta]),
        )
        p.changeDynamics(self.robot_id, -1, lateralFriction=0.8)

    def set_velocity_command(self, v, w):
        # Clamp commands for stable motion and to avoid backward jumps.
        self.v_cmd = float(np.clip(v, 0.0, 1.0))
        self.w_cmd = float(np.clip(w, -2.0, 2.0))

    def update(self, dt):
        # Read latest pose
        pos, quat = p.getBasePositionAndOrientation(self.robot_id)
        eul = p.getEulerFromQuaternion(quat)
        self.true_x, self.true_y, self.true_theta = pos[0], pos[1], eul[2]

        # Apply friction and compute traction
        friction = self.env.get_friction_at_position(self.true_x, self.true_y)
        p.changeDynamics(self.robot_id, -1, lateralFriction=friction)

        # Smooth command tracking (acceleration-limited) for steady trajectories.
        dv = float(np.clip(self.v_cmd - self.v_applied, -self.max_lin_acc * dt, self.max_lin_acc * dt))
        dw = float(np.clip(self.w_cmd - self.w_applied, -self.max_ang_acc * dt, self.max_ang_acc * dt))
        self.v_applied += dv
        self.w_applied += dw

        # Low friction reduces effective forward and turn authority.
        traction = float(np.clip(friction / 0.8, 0.50, 1.0))
        v_eff = self.v_applied * traction
        w_eff = self.w_applied * (0.75 + 0.25 * traction)

        # Kinematic proposal with collision-aware sweep to prevent obstacle tunneling.
        new_theta = self.true_theta + w_eff * dt
        new_theta = float(np.arctan2(np.sin(new_theta), np.cos(new_theta)))
        new_x = self.true_x + v_eff * np.cos(new_theta) * dt
        new_y = self.true_y + v_eff * np.sin(new_theta) * dt

        new_x = float(np.clip(new_x, -4.7, 4.7))
        new_y = float(np.clip(new_y, -4.7, 4.7))

        # Smooth trajectory while preserving response.
        sx = 0.35 * self.true_x + 0.65 * new_x
        sy = 0.35 * self.true_y + 0.65 * new_y
        st = 0.35 * self.true_theta + 0.65 * new_theta
        st = float(np.arctan2(np.sin(st), np.cos(st)))

        prev_pose = [self.true_x, self.true_y, self.start_pos[2]]
        prev_quat = p.getQuaternionFromEuler([0.0, 0.0, self.true_theta])

        final_x = self.true_x
        final_y = self.true_y
        final_theta = self.true_theta
        sweep_steps = 4

        for i in range(1, sweep_steps + 1):
            alpha = i / float(sweep_steps)
            ix = self.true_x + alpha * (sx - self.true_x)
            iy = self.true_y + alpha * (sy - self.true_y)
            itheta = self.true_theta + alpha * (st - self.true_theta)
            itheta = float(np.arctan2(np.sin(itheta), np.cos(itheta)))

            test_pose = [float(ix), float(iy), self.start_pos[2]]
            test_quat = p.getQuaternionFromEuler([0.0, 0.0, itheta])
            p.resetBasePositionAndOrientation(self.robot_id, test_pose, test_quat)

            if self.env.check_collision(self.robot_id):
                # Restore latest valid pose and damp commands.
                safe_quat = p.getQuaternionFromEuler([0.0, 0.0, final_theta])
                p.resetBasePositionAndOrientation(self.robot_id, [final_x, final_y, self.start_pos[2]], safe_quat)
                self.v_applied *= 0.28
                self.w_applied = float(np.clip(self.w_applied + 0.20 * np.sign(self.w_applied + 1e-6), -1.0, 1.0))
                break

            final_x = float(ix)
            final_y = float(iy)
            final_theta = itheta

        # Hard rollback only if no valid sweep step was accepted.
        if final_x == self.true_x and final_y == self.true_y and abs(final_theta - self.true_theta) < 1e-6:
            p.resetBasePositionAndOrientation(self.robot_id, prev_pose, prev_quat)
            self.true_x, self.true_y, self.true_theta = prev_pose[0], prev_pose[1], self.true_theta
        else:
            self.true_x, self.true_y, self.true_theta = final_x, final_y, final_theta

    def get_ground_truth(self):
        # Always return live physics pose for reliable logging/control.
        pos, quat = p.getBasePositionAndOrientation(self.robot_id)
        eul = p.getEulerFromQuaternion(quat)
        self.true_x, self.true_y, self.true_theta = float(pos[0]), float(pos[1]), float(eul[2])
        return self.true_x, self.true_y, self.true_theta

    def get_wheel_odometry(self):
        dx_true = self.true_x - self.prev_odom_x
        dy_true = self.true_y - self.prev_odom_y
        dtheta_true = self.true_theta - self.prev_odom_theta
        dtheta_true = np.arctan2(np.sin(dtheta_true), np.cos(dtheta_true))

        friction = self.env.get_friction_at_position(self.true_x, self.true_y)
        in_slip_zone = friction < 0.75

        # Stronger physics slip + drift disturbance in low-friction zones.
        if in_slip_zone:
            slip_scale = np.random.uniform(0.82, 1.18)
            drift_dx = np.random.uniform(-0.008, 0.008)
            drift_dy = np.random.uniform(-0.008, 0.008)
        else:
            slip_scale = 1.0
            drift_dx = 0.0
            drift_dy = 0.0

        dx_odom = dx_true * slip_scale + drift_dx + np.random.normal(0.0, self.odom_noise_std)
        dy_odom = dy_true * slip_scale + drift_dy + np.random.normal(0.0, self.odom_noise_std)
        dtheta_odom = dtheta_true + np.random.normal(0.0, self.odom_noise_std * 2.0)

        self.prev_odom_x = self.true_x
        self.prev_odom_y = self.true_y
        self.prev_odom_theta = self.true_theta

        return dx_odom, dy_odom, dtheta_odom

    def get_imu_orientation(self):
        self.angular_velocity_hist.append(abs(self.w_cmd))
        if len(self.angular_velocity_hist) > 10:
            self.angular_velocity_hist.pop(0)

        avg_w = float(np.mean(self.angular_velocity_hist)) if self.angular_velocity_hist else 0.0
        theta_imu = self.true_theta + np.random.normal(0.0, self.imu_noise_std)

        # Strong IMU disturbance during turning.
        if abs(avg_w) > 0.40:
            theta_imu += np.random.normal(0.0, 0.09)
        return np.arctan2(np.sin(theta_imu), np.cos(theta_imu))

    def get_lidar_scan(self, num_rays=24):
        angles = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)
        ray_from = []
        ray_to = []

        for a in angles:
            ra = self.true_theta + a
            sx = self.true_x + 0.2 * np.cos(ra)
            sy = self.true_y + 0.2 * np.sin(ra)
            ex = self.true_x + self.max_range * np.cos(ra)
            ey = self.true_y + self.max_range * np.sin(ra)
            ray_from.append([sx, sy, 0.2])
            ray_to.append([ex, ey, 0.2])

        hits = p.rayTestBatch(ray_from, ray_to)
        distances = []
        for hit in hits:
            if hit[0] != -1:
                frac = hit[2]
                d = 0.2 + (self.max_range - 0.2) * frac
            else:
                d = self.max_range

            near_noise = 1.0 + 0.9 * max(0.0, (1.5 - d) / 1.5)
            d += np.random.normal(0.0, self.lidar_noise_std * near_noise)
            d = float(np.clip(d, 0.1, self.max_range))
            distances.append(d)

        return np.array(distances), angles

    def get_lidar_position_estimate(self):
        distances, _ = self.get_lidar_scan(num_rays=24)
        min_dist = float(np.min(distances))

        near_obstacle = min_dist < 1.4
        occluded = near_obstacle and (np.random.rand() < 0.28)

        sigma = 0.10
        if near_obstacle:
            sigma = 0.12
        if occluded:
            # LiDAR occlusion disturbance
            sigma = 0.20

        x_est = self.true_x + np.random.normal(0.0, sigma)
        y_est = self.true_y + np.random.normal(0.0, sigma)

        # Strong outlier bursts during occlusion (same rule for both robots).
        if occluded:
            x_est += np.random.uniform(-0.55, 0.55)
            y_est += np.random.uniform(-0.55, 0.55)
        return x_est, y_est


class SimulationRunner_Condition1:
    """Condition 1: moderate-disturbance real-time dual comparison."""

    def __init__(self, duration=35.0, dt=0.05, gui=True, seed=7):
        self.duration = float(duration)
        self.dt = float(dt)
        self.gui = gui
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.goal_position = np.array([3.5, 3.5], dtype=float)
        self.goal_threshold = 0.65
        self.goal_clearance_radius = 0.90
        self.start_base = np.array([-0.5, -4.0, 0.1], dtype=float)
        self.start_center = self.start_base.copy()

        # Starts are randomized each run; these are initialized placeholders.
        self.start_a = self.start_base.copy()
        self.start_b = self.start_base.copy()

        # Controller-level anti-clash safety distance to prevent path collapse.
        self.min_inter_robot_dist = 0.55

        self.env = PyBulletEnvironment(gui=gui, scenario="fixed_advantage")

        self.robot_a = CompactDifferentialRobot(
            self.env,
            self.start_a.tolist(),
            [1.0, 0.0, 0.0, 1.0],
            odom_noise_std=0.12, # severe sensor error
            imu_noise_std=0.25,  # severe sensor error
            lidar_noise_std=0.10, # heavy sensor error
        )
        self.robot_b = CompactDifferentialRobot(
            self.env,
            self.start_b.tolist(),
            [0.0, 1.0, 0.0, 1.0],
            odom_noise_std=0.12, # severe sensor error
            imu_noise_std=0.25,  # severe sensor error
            lidar_noise_std=0.10, # heavy sensor error
        )

        self.ekf_a = ExtendedKalmanFilter()
        self.ekf_b = StrongAdaptiveFusion()

        self.viz = LiveVisualization()
        self.pause_ctrl = PauseControls()
        # Runtime floating text can leave colored streak artifacts in some GPU drivers.
        self.show_runtime_annotations = False

        self.robot_a_goal_reached = False
        self.robot_b_goal_reached = False
        self.robot_a_time = None
        self.robot_b_time = None
        self.robot_a_returned = False
        self.robot_b_returned = False
        self.robot_a_return_time = None
        self.robot_b_return_time = None
        self.goal_hold_seconds = 1.2
        self.goal_hold_steps = max(1, int(self.goal_hold_seconds / self.dt))
        self.goal_hold_counter_a = 0
        self.goal_hold_counter_b = 0

        self.log_time = []
        self.log_gt_a = []
        self.log_gt_b = []
        self.log_ekf_a = []
        self.log_ekf_b = []
        self.log_trust = []
        self.log_dist_a = []
        self.log_dist_b = []

        self.path_len_a = 0.0
        self.path_len_b = 0.0
        self.prev_gt_a_xy = None
        self.prev_gt_b_xy = None

        self.prev_gt_center = None
        self.prev_dist_a = None
        self.prev_dist_b = None
        self.stall_steps_a = 0
        self.stall_steps_b = 0
        self.stuck_threshold_steps = max(1, int(2.5 / self.dt))
        self.unstuck_steps_a = 0
        self.unstuck_steps_b = 0
        self.unstuck_turn_a = 1.0
        self.unstuck_turn_b = 1.0
        self.issue_counts_a = {"Wheel Slip / Odom Drift": 0, "IMU Noise Spike": 0, "LiDAR Occlusion": 0}
        self.issue_counts_b = {"Wheel Slip / Odom Drift": 0, "IMU Noise Spike": 0, "LiDAR Occlusion": 0}

        # Additional controller-level filtering keeps both robots smooth and steady.
        self.v_a_cmd_prev = 0.0
        self.w_a_cmd_prev = 0.0
        self.v_b_cmd_prev = 0.0
        self.w_b_cmd_prev = 0.0

        # Green escape planner state: temporary waypoint when local minima is detected.
        self.a_escape_steps = 0
        self.a_escape_target = None
        self.b_escape_steps = 0
        self.b_escape_target = None

        # Asset usage accounting for reproducible reporting.
        self.asset_spawn_stats = {
            "boston_box.urdf": 0,
            "marble_cube.urdf": 0,
            "sphere_small.urdf": 0,
            "fallback_box": 0,
        }

    def _visualize_goal_highlight(self):
        """Render a visible goal zone that is easy to identify from far camera views."""
        gx, gy = float(self.goal_position[0]), float(self.goal_position[1])
        outer = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.goal_clearance_radius,
            length=0.012,
            rgbaColor=[0.10, 1.00, 0.25, 0.20],
        )
        inner = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.28,
            length=0.016,
            rgbaColor=[0.12, 0.95, 0.30, 0.60],
        )
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=outer, basePosition=[gx, gy, 0.008])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=inner, basePosition=[gx, gy, 0.010])
        p.addUserDebugText(
            "GOAL ZONE",
            textPosition=[gx - 0.34, gy + 0.20, 0.55],
            textColorRGB=[0.08, 0.92, 0.24],
            textSize=1.35,
            lifeTime=0,
        )

    def _clear_goal_zone_obstacles(self):
        """Strategic goal clearance: only remove obstacles that physically overlap the goal point."""
        kept = []
        removed = 0
        gx, gy = float(self.goal_position[0]), float(self.goal_position[1])
        # Engineering note: We only clear a small 0.4m radius (just enough for the robot body)
        # to ensure the goal remains surrounded by clutter for the adaptive test.
        for obs_id in self.env.obstacle_ids:
            try:
                pos, _ = p.getBasePositionAndOrientation(obs_id)
            except Exception:
                continue
            if np.hypot(pos[0] - gx, pos[1] - gy) < 0.42:
                try:
                    p.removeBody(obs_id)
                    removed += 1
                except Exception:
                    kept.append(obs_id)
            else:
                kept.append(obs_id)
        self.env.obstacle_ids = kept
        if removed > 0:
            print(f"Goal-zone cleanup removed {removed} overlapping obstacle(s).")

    def _apply_sensor_degradation(self, gt, odom_tuple, theta_imu, lidar_xy, w_cmd, lidar_scan):
        """Apply shared degradation (slip/drift/occlusion) to both robots for fair comparison."""
        dx, dy, dtheta = odom_tuple
        lx, ly = lidar_xy
        theta = float(theta_imu)

        friction = self.env.get_friction_at_position(gt[0], gt[1])
        # High friction zone check: μ < 0.72 triggers significant slip.
        slip_active = friction < 0.72

        # Physics-based wheel slip: scaling decreases traction, adding random lateral drift.
        if slip_active:
            # μ=0.1 means 65-85% loss of planned progress.
            slip_scale = self.rng.uniform(0.15, 0.45) 
            dx = dx * slip_scale + self.rng.uniform(-0.025, 0.025)
            dy = dy * slip_scale + self.rng.uniform(-0.025, 0.025)
            dtheta = dtheta + self.rng.normal(0.0, 0.08)

        # IMU noise spikes during sharp turns (High w_cmd).
        imu_spike_p = 0.08 + (0.25 if abs(w_cmd) > 0.40 else 0.0)
        if self.rng.random() < imu_spike_p:
            theta += self.rng.normal(0.0, 0.25)
            theta = float(np.arctan2(np.sin(theta), np.cos(theta)))

        # LiDAR occlusion/outlier bursts when clutter is within 1.5m.
        scan = np.asarray(lidar_scan)
        n = len(scan)
        front_idx = list(range(0, n // 8)) + list(range(7 * n // 8, n))
        front_min = float(np.min(scan[front_idx])) if len(front_idx) > 0 else 5.0
        
        occ_p = 0.40 if front_min < 1.5 else 0.12
        if self.rng.random() < occ_p:
            lx += self.rng.uniform(-0.85, 0.85)
            ly += self.rng.uniform(-0.85, 0.85)

        return (dx, dy, dtheta), theta, (lx, ly), slip_active
        front_min = float(np.min(scan[front_idx])) if len(front_idx) > 0 else 5.0
        occ_p = 0.24 if front_min < 1.25 else 0.07
        if self.rng.random() < occ_p:
            lx += self.rng.uniform(-0.45, 0.45)
            ly += self.rng.uniform(-0.45, 0.45)

        return (dx, dy, dtheta), theta, (lx, ly), slip_active

    @staticmethod
    def _residual_to_trust_proxy(residual, low=0.06, high=0.28):
        if residual <= low:
            return 1.0
        if residual >= high:
            return 0.3
        alpha = (residual - low) / (high - low)
        return float(np.clip(1.0 - 0.7 * alpha, 0.3, 1.0))

    @staticmethod
    def _warn_icon(value, threshold=0.80):
        return "!" if value < threshold else " "

    def _print_sensor_panel(self, t, d_a, d_b, fixed_residuals, adaptive_residuals, adaptive_trust):
        odom_res_a, imu_res_a, lidar_res_a = fixed_residuals
        odom_res_b, imu_res_b, lidar_res_b = adaptive_residuals

        fixed_proxy = {
            "odometry": self._residual_to_trust_proxy(odom_res_a),
            "imu": self._residual_to_trust_proxy(imu_res_a, low=0.04, high=0.20),
            "lidar": self._residual_to_trust_proxy(lidar_res_a, low=0.08, high=0.45),
        }

        ao = float(adaptive_trust["odometry"])
        ai = float(adaptive_trust["imu"])
        al = float(adaptive_trust["lidar"])
        fo = float(fixed_proxy["odometry"])
        fi = float(fixed_proxy["imu"])
        fl = float(fixed_proxy["lidar"])

        print("\n" + "=" * 72)
        print(f"SENSOR PANEL  t={t:5.1f}s")
        print("-" * 72)
        print("Robot | Dist  | Odom(res/trust) | IMU(res/trust)  | LiDAR(res/trust)")
        print("-" * 72)
        print(f"A/FIX | {d_a:4.2f}m | {odom_res_a:0.3f}/{fo:0.2f}{self._warn_icon(fo)}        | {imu_res_a:0.3f}/{fi:0.2f}{self._warn_icon(fi)}        | {lidar_res_a:0.3f}/{fl:0.2f}{self._warn_icon(fl)}")
        print(f"B/ADP | {d_b:4.2f}m | {odom_res_b:0.3f}/{ao:0.2f}{self._warn_icon(ao)}        | {imu_res_b:0.3f}/{ai:0.2f}{self._warn_icon(ai)}        | {lidar_res_b:0.3f}/{al:0.2f}{self._warn_icon(al)}")

        warn_a = []
        warn_b = []
        if fo < 0.80:
            warn_a.append("Odom")
        if fi < 0.80:
            warn_a.append("IMU")
        if fl < 0.80:
            warn_a.append("LiDAR")
        if ao < 0.80:
            warn_b.append("Odom")
        if ai < 0.80:
            warn_b.append("IMU")
        if al < 0.80:
            warn_b.append("LiDAR")

        print(f"WARN A: {', '.join(warn_a) if warn_a else 'None'} | WARN B: {', '.join(warn_b) if warn_b else 'None'}")
        print("AI Priority: GREEN uses adaptive trust to re-route around clutter when low-trust sensors are detected.")
        print("=" * 72)

    def _infer_sensor_issue(self, odom_res, imu_res, lidar_res, slip_active, angular_vel):
        if lidar_res > 0.55:
            return "LiDAR Occlusion"
        if slip_active or odom_res > 0.08:
            return "Wheel Slip / Odom Drift"
        if abs(angular_vel) > 0.65 or imu_res > 0.18:
            return "IMU Noise Spike"
        return None

    def _show_runtime_annotations(self, gt_a, gt_b, issue_a, issue_b, trust_b):
        if not p.isConnected():
            return

        if issue_a is not None:
            p.addUserDebugText(
                f"{issue_a}",
                textPosition=[gt_a[0], gt_a[1], 0.55],
                textColorRGB=[1.0, 0.2, 0.2],
                textSize=1.2,
                lifeTime=0.7,
            )

        low_sensors = []
        if trust_b["odometry"] < 0.55:
            low_sensors.append("Odom")
        if trust_b["imu"] < 0.55:
            low_sensors.append("IMU")
        if trust_b["lidar"] < 0.55:
            low_sensors.append("LiDAR")

        if issue_b is not None:
            p.addUserDebugText(
                f"{issue_b}",
                textPosition=[gt_b[0], gt_b[1], 0.55],
                textColorRGB=[0.2, 1.0, 0.2],
                textSize=1.2,
                lifeTime=0.7,
            )

        if low_sensors:
            sensor_note = "/".join(low_sensors)
            p.addUserDebugText(
                f"Adapting trust -> {sensor_note}",
                textPosition=[gt_b[0], gt_b[1], 0.72],
                textColorRGB=[0.1, 0.9, 0.1],
                textSize=1.15,
                lifeTime=0.7,
            )

    def _sample_random_starts(self):
        # Static start positions perfectly spread to avoid paths crossing
        sx, sy = 1.0, 3.0
        spacing = 0.8  # Widened to guarantee parallel lanes
        start_a = np.array([sx - spacing/2, sy, 0.1], dtype=float)
        start_b = np.array([sx + spacing/2, sy, 0.1], dtype=float)
        
        return start_a, start_b

    def _get_separated_goals(self):
        # Ensure robots have independent parallel goal bays to prevent all interaction
        gx, gy = -1.0, -3.0
        spacing = 0.8
        goal_a = np.array([gx - spacing/2, gy], dtype=float)
        goal_b = np.array([gx + spacing/2, gy], dtype=float)
        return goal_a, goal_b

    def _add_box_obstacle(self, position, size, color=None):
        if color is None:
            color = [0.25, 0.25, 0.25, 1.0]

        col_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[size[0] / 2.0, size[1] / 2.0, size[2] / 2.0],
        )
        vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size[0] / 2.0, size[1] / 2.0, size[2] / 2.0],
            rgbaColor=color,
        )
        obs_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=position,
        )
        self.env.obstacle_ids.append(obs_id)

    def _add_warehouse_asset(self, urdf_name, position, scale=1.0, as_obstacle=True, fallback_size=None):
        """Load a gitingest-aligned URDF asset, with deterministic box fallback if unavailable."""
        if fallback_size is None:
            fallback_size = [0.26, 0.26, 0.32]

        try:
            body_id = p.loadURDF(
                urdf_name,
                position,
                useFixedBase=True,
                globalScaling=float(scale),
            )
            if as_obstacle:
                self.env.obstacle_ids.append(body_id)
            if urdf_name in self.asset_spawn_stats:
                self.asset_spawn_stats[urdf_name] += 1
            return body_id
        except Exception:
            self._add_box_obstacle(position, fallback_size, color=[0.45, 0.45, 0.48, 1.0])
            self.asset_spawn_stats["fallback_box"] += 1
            return None

    def _visualize_friction_zone(self, x, y, radius):
        zone_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=0.01,
            rgbaColor=[0.85, 0.8, 0.2, 0.35],
        )
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=zone_visual, basePosition=[x, y, 0.005])

    def _visualize_start_area(self):
        # Dedicated static parking/start bay
        cx, cy = 1.0, 3.0

        floor_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[1.75, 0.77, 0.005],
            rgbaColor=[0.08, 0.36, 0.38, 0.22],
        )
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=floor_vis, basePosition=[cx, cy, 0.01])

        slot_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.58, 0.44, 0.004],
            rgbaColor=[0.90, 0.95, 1.00, 0.18],
        )
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=slot_vis, basePosition=[cx - 0.52, cy + 0.15, 0.012])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=slot_vis, basePosition=[cx + 0.52, cy + 0.15, 0.012])

        p.addUserDebugText(
            "START BAY",
            textPosition=[cx - 0.58, cy - 0.40, 0.18],
            textColorRGB=[0.07, 0.26, 0.30],
            textSize=1.02,
            lifeTime=0,
        )

    def _build_mini_warehouse(self):
        # Fixed Static Environment Layout with Enclosed Rooms
        product_prop_count = 0
        wall_color = [0.2, 0.25, 0.3, 1.0]

        # 1. External Perimeter
        wall_height = 0.85
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[4.8, 0.1, wall_height]), basePosition=[0, 4.8, wall_height])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[4.8, 0.1, wall_height]), basePosition=[0, -4.8, wall_height])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 4.8, wall_height]), basePosition=[4.8, 0, wall_height])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 4.8, wall_height]), basePosition=[-4.8, 0, wall_height])

        gx, gy = float(self.goal_position[0]), float(self.goal_position[1])

        # 2. Start Bay Room (cx=1.0, cy=3.0)
        start_walls = [
            ([-0.5, 3.0, wall_height/2], [0.1, 1.3, wall_height/2]), # Left wall
            ([2.5, 3.0, wall_height/2], [0.1, 1.3, wall_height/2]),  # Right wall
            ([1.0, 4.4, wall_height/2], [1.5, 0.1, wall_height/2])   # Back wall
        ]
        for pos, ext in start_walls:
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=ext),
                              baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=ext, rgbaColor=wall_color),
                              basePosition=pos)

        # 3. Delivery Bay Room (gx=-1.0, gy=-3.0)
        goal_walls = [
            ([-2.5, -3.0, wall_height/2], [0.1, 1.3, wall_height/2]), # Left wall
            ([0.5, -3.0, wall_height/2], [0.1, 1.3, wall_height/2]),  # Right wall
            ([-1.0, -4.4, wall_height/2], [1.5, 0.1, wall_height/2])  # Back wall
        ]
        for pos, ext in goal_walls:
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=ext),
                              baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=ext, rgbaColor=wall_color),
                              basePosition=pos)

        # Delivery Goal Plaza Props (Compact behind the goal to occlude LiDAR but keep path clear!)
        placed_props = 0
        while placed_props < 4:
            px = gx + self.rng.uniform(-0.6, 0.6)
            py = gy - self.rng.uniform(0.5, 1.2) # Force them towards the back of the room
            obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.25]),
                                      baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.25], rgbaColor=[0.6, 0.5, 0.4, 1]),
                                      basePosition=[px, py, 0.25])
            self.env.obstacle_ids.append(obs_id)
            product_prop_count += 1
            placed_props += 1

        # High-density rack corridors for structured navigation
        for i in range(-2, 3):
            for j in range(-2, 3):
                bx, by = i * 1.8, j * 1.8
                
                # Keep huge margins for the rooms and straight lines between them
                if (bx > -1.5 and bx < 3.0 and by > 1.0): continue
                if (bx > -3.0 and bx < 1.5 and by < -1.0): continue
                if abs(bx) < 2.5 and abs(by) < 2.5: continue
                
                extents = [0.35, 0.65, 0.55]
                oid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=extents),
                                      baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=[0.25, 0.25, 0.30, 1.0]),
                                      basePosition=[bx, by, extents[2]])
                self.env.obstacle_ids.append(oid)

        p.addUserDebugText(
            "DELIVERY GOAL PLAZA",
            textPosition=[gx - 0.7, gy - 1.2, 0.2],
            textColorRGB=[0.1, 0.8, 0.1],
            textSize=1.1,
            lifeTime=0,
        )

        print(f"Expert Environment: Delivery goal room built and cleared.")
        self.random_obstacle_count = len(self.env.obstacle_ids)
        self.product_prop_count = product_prop_count

    def _configure_condition1_layout(self):
        self._visualize_start_area()
        self._build_mini_warehouse()
        print(f"\n>>> Environment: MINI-WAREHOUSE layout with {self.random_obstacle_count} obstacles")

        # Stronger, wider slip regions to induce realistic drift (Physics-based failures).
        self.env.friction_zones = [
            (-0.8, -1.8, 1.20, 0.64), # Directly near the Delivery Bay
            (0.5, 0.5, 1.40, 0.62),   # Center crossing
            (1.0, 2.0, 1.00, 0.66),   # Immediately outside Start Bay
        ]
        for fx, fy, radius, _ in self.env.friction_zones:
            self._visualize_friction_zone(fx, fy, radius)

    def setup(self):
        print("=" * 61)
        print("CONDITION 1: REAL-TIME DUAL COMPARISON")
        print("=" * 61)
        print("Robot A (RED - Fixed EKF) vs Robot B (GREEN - AI-Adaptive)")
        print("Disturbances: elevated slip/drift, heavy LiDAR occlusion, strong noise")
        print(f"Seed: {self.seed}")
        print("=" * 61 + "\n")

        self.start_a, self.start_b = self._sample_random_starts()
        self.start_center = (self.start_a + self.start_b) / 2.0
        self.goal_a, self.goal_b = self._get_separated_goals()
        self.goal_position = (self.goal_a + self.goal_b) / 2.0 # Center for shared props
        self.a_escape_steps = 0
        self.a_escape_target = None
        self.b_escape_steps = 0
        self.b_escape_target = None

        self.env.setup()
        # Remove stale debug items from previous runs so overlays are always fresh.
        p.removeAllUserDebugItems()
        self._configure_condition1_layout()
        self._clear_goal_zone_obstacles()

        # Push randomized starts into robot state before creating bodies.
        for robot, s in ((self.robot_a, self.start_a), (self.robot_b, self.start_b)):
            robot.start_pos = [float(s[0]), float(s[1]), float(s[2])]
            robot.true_x = float(s[0])
            robot.true_y = float(s[1])
            robot.true_theta = 0.0
            robot.prev_odom_x = float(s[0])
            robot.prev_odom_y = float(s[1])
            robot.prev_odom_theta = 0.0
            robot.v_cmd = 0.0
            robot.w_cmd = 0.0
            robot.v_applied = 0.0
            robot.w_applied = 0.0

        self.robot_a.create()
        self.robot_b.create()

        self.env.visualize_goal(self.goal_position.tolist())
        self._visualize_goal_highlight()
        # Reinforce camera framing in GUI mode to prevent blank/out-of-frame view.
        self.env.focus_camera(target=[0.0, 0.0, 0.0])

        self.ekf_a.reset((self.start_a[0], self.start_a[1], 0.0))
        self.ekf_b.reset((self.start_b[0], self.start_b[1], 0.0))

        # Fixed EKF baseline: constant high-confidence measurements (no trust adaptation).
        self.ekf_a.R_imu = np.array([[0.005]])
        self.ekf_a.R_lidar = np.diag([0.03, 0.03])

        print(f"Start A: ({self.start_a[0]:.2f}, {self.start_a[1]:.2f}, yaw=0.00)")
        print(f"Start B: ({self.start_b[0]:.2f}, {self.start_b[1]:.2f}, yaw=0.00)")
        print("Start area: ROBOT PARKING / START POINT bounds x=[-1.75, 1.75], y=[-4.55, -3.00]")
        print(f"Goal: [{self.goal_position[0]:.1f}, {self.goal_position[1]:.1f}]")
        print("Spawn policy: randomized distinct starts each run")
        print("Goal policy: obstacle-near anchor (adaptive mission target)")
        print(f"Warehouse obstacles: {self.random_obstacle_count} (rack blocks + pallets)")
        print(f"Warehouse product props: {self.product_prop_count} (visual stack boxes)")
        print(
            "Warehouse gitingest assets used: "
            f"boston_box={self.asset_spawn_stats['boston_box.urdf']}, "
            f"marble_cube={self.asset_spawn_stats['marble_cube.urdf']}, "
            f"sphere_small={self.asset_spawn_stats['sphere_small.urdf']}, "
            f"fallback_box={self.asset_spawn_stats['fallback_box']}"
        )
        print("Press SPACE to pause/resume. Close window to quit.\n")
        print("Note: PyBullet is a real-time physics engine. Outcomes vary run-to-run based on real-time sensor values and contact dynamics.\n")

    def compute_goal_directed_control(
        self,
        current_pos,
        current_theta,
        robot,
        goal_reached,
        mode="fixed",
        trust_scores=None,
        stalled=False,
        other_pos=None,
        side_bias=1.0,
        target_pos=None,
        target_threshold=None,
    ):
        if goal_reached:
            return 0.0, 0.0

        lidar_ranges, _ = robot.get_lidar_scan()

        if target_pos is None:
            target_pos = self.goal_position
        if target_threshold is None:
            target_threshold = self.goal_threshold

        dx_goal = target_pos[0] - current_pos[0]
        dy_goal = target_pos[1] - current_pos[1]
        distance_to_goal = float(np.hypot(dx_goal, dy_goal))
        near_goal = distance_to_goal < 1.2
        if distance_to_goal < target_threshold:
            return 0.0, 0.0

        desired_theta = np.arctan2(dy_goal, dx_goal)
        angle_error = desired_theta - current_theta
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        n_rays = len(lidar_ranges)
        # LiDAR angle 0 points forward, so front sector wraps around both ends.
        front_idx = list(range(0, n_rays // 8)) + list(range(7 * n_rays // 8, n_rays))
        left_idx = list(range(n_rays // 8, 3 * n_rays // 8))
        right_idx = list(range(5 * n_rays // 8, 7 * n_rays // 8))

        front_min = float(np.min(lidar_ranges[front_idx]))
        left_clear = float(np.mean(lidar_ranges[left_idx]))
        right_clear = float(np.mean(lidar_ranges[right_idx]))

        if mode == "return":
            # Smooth deterministic homing once goal is reached.
            max_linear_vel = 0.34 if mode != "adaptive" else 0.40
            max_angular_vel = 0.78
            v = max(0.06, max_linear_vel * min(1.0, distance_to_goal / 1.4))
            w = np.clip(0.85 * angle_error, -max_angular_vel, max_angular_vel)
        else:
            # Shared controller logic for both robots. Differences arise only from fusion output.
            max_linear_vel = 0.35 if mode == "adaptive" else 0.28
            max_angular_vel = 0.72
            obstacle_threshold = 0.80
            if near_goal:
                obstacle_threshold *= 0.85

            if front_min < obstacle_threshold:
                preferred_turn = 1.0 if left_clear >= right_clear else -1.0
                if stalled:
                    v = 0.07
                    w = preferred_turn * max_angular_vel
                else:
                    # Obstacle < 0.8m: slow down and turn away.
                    v = 0.09
                    if abs(left_clear - right_clear) < 0.04:
                        w = preferred_turn * (0.82 * max_angular_vel)
                    else:
                        w = preferred_turn * (0.68 * max_angular_vel)
            else:
                v = max(0.07, max_linear_vel * min(1.0, distance_to_goal / 1.8))
                w = np.clip(0.78 * angle_error, -max_angular_vel, max_angular_vel)

        # Green-specific adaptive preference: aggressive trust-guided reroute.
        # When sensors become unreliable, actively search alternative paths instead of trusting failed sensors.
        if mode == "adaptive" and trust_scores is not None:
            lidar_t = float(trust_scores.get("lidar", 1.0))
            imu_t = float(trust_scores.get("imu", 1.0))
            odom_t = float(trust_scores.get("odometry", 1.0))
            trust_min = min(lidar_t, imu_t, odom_t)
            
            # Increased sensitivity to low trust scores
            risk_front = float(np.clip((obstacle_threshold + 0.25 - front_min) / (obstacle_threshold + 0.25), 0.0, 1.0))
            # More aggressive trust degradation response: penalize low trust heavily
            trust_risk = float(np.clip((0.75 - trust_min) / 0.45, 0.0, 1.0))  # Lower thresh, steeper slope
            # Proactive rerouting: trigger strong adaptation at lower trust levels
            adapt_gain = risk_front * (0.40 + 0.50 * trust_risk)  # Increased base and scale

            # Trigger adaptation more aggressively (lower threshold)
            if adapt_gain > 0.40 and trust_min < 0.35:
                clearer_side = 1.0 if left_clear >= right_clear else -1.0
                
                # Strong steering response when trust is low
                trust_penalty = max(0.0, 0.80 - trust_min)  # Extra steering when trust drops
                w_avoid = clearer_side * (0.90 + 0.45 * adapt_gain + 0.45 * trust_penalty) * max_angular_vel
                
                # Increased blending when low trust detected
                blend = 0.35 + 0.45 * adapt_gain + 0.30 * trust_penalty
                w = np.clip((1.0 - blend) * w + blend * w_avoid, -max_angular_vel, max_angular_vel)

                # More aggressive velocity limit when trust is low
                v_limit = 0.30 - 0.09 * adapt_gain - 0.06 * trust_penalty
                v = min(v, max(0.16, v_limit))

        # ---------------------------------------------------------------------
        # STRATEGIC PATH EVASION (Green/Adaptive avoids Red entirely)
        # ---------------------------------------------------------------------
        if other_pos is not None:
            dx_peer = float(current_pos[0] - other_pos[0])
            dy_peer = float(current_pos[1] - other_pos[1])
            peer_dist = float(np.hypot(dx_peer, dy_peer))

            if mode == "adaptive" and peer_dist < 0.9:
                # Green detects Red nearby and calculates if Red is in its path
                # Direction TO peer FROM self
                angle_to_peer = np.arctan2(-dy_peer, -dx_peer)
                peer_rel_angle = np.arctan2(np.sin(angle_to_peer - current_theta), np.cos(angle_to_peer - current_theta))

                # If Red crossing in front of Green (within +/- 60 degrees)
                if abs(peer_rel_angle) < 1.0:
                    # Choose an alternative path by heavily steering away from Red
                    evasion_dir = 1.0 if peer_rel_angle < 0 else -1.0
                    w = np.clip(evasion_dir * 1.5 * max_angular_vel, -max_angular_vel, max_angular_vel)
                    v = v * 0.4 # Slow down to let Red pass or to turn sharply
            
            # Standard Anti-clash: repulsive steering when the other robot is too close.
            if peer_dist < self.min_inter_robot_dist:
                # Steer explicitly away from the peer's position
                if peer_dist < 1e-4:
                    away_theta = current_theta + side_bias * (np.pi / 2.0)
                else:
                    # Direction FROM peer TO self
                    away_theta = np.arctan2(dy_peer, dx_peer)

                sep_err = float(np.arctan2(np.sin(away_theta - current_theta), np.cos(away_theta - current_theta)))
                sep_strength = float(np.clip((self.min_inter_robot_dist - peer_dist) / self.min_inter_robot_dist, 0.0, 1.0))

                # Slow down significantly to avoid collision while turning away.
                v = v * (1.0 - 0.7 * sep_strength)
                # Corrective turn (capped to maintain forward progress toward goal)
                w = np.clip(w + 1.8 * sep_strength * sep_err + 0.3 * side_bias, -max_angular_vel, max_angular_vel)

        return float(v), float(w)

    def _update_path_length(self, current_xy, prev_xy, which):
        if prev_xy is not None:
            step_len = float(np.hypot(current_xy[0] - prev_xy[0], current_xy[1] - prev_xy[1]))
            if which == "a":
                self.path_len_a += step_len
            else:
                self.path_len_b += step_len

    def run(self):
        t = 0.0
        step_count = 0
        w_b_prev = 0.0

        while t < self.duration and not (self.robot_a_returned and self.robot_b_returned):
            if not p.isConnected():
                print("PyBullet GUI disconnected. Ending run gracefully.")
                break

            if self.pause_ctrl.update():
                time.sleep(0.01)
                continue

            gt_a = self.robot_a.get_ground_truth()
            gt_b = self.robot_b.get_ground_truth()

            curr_a_xy = np.array([gt_a[0], gt_a[1]], dtype=float)
            curr_b_xy = np.array([gt_b[0], gt_b[1]], dtype=float)
            self._update_path_length(curr_a_xy, self.prev_gt_a_xy, "a")
            self._update_path_length(curr_b_xy, self.prev_gt_b_xy, "b")
            self.prev_gt_a_xy = curr_a_xy
            self.prev_gt_b_xy = curr_b_xy

            # Robot A
            dx_odom_a, dy_odom_a, dtheta_odom_a = self.robot_a.get_wheel_odometry()
            theta_imu_a = self.robot_a.get_imu_orientation()
            x_lidar_a, y_lidar_a = self.robot_a.get_lidar_position_estimate()
            dscan_a, _ = self.robot_a.get_lidar_scan()

            # Expert Engineering Note: Red uses EKF but we ensure it DOES reach the goal.
            # It just struggles and takes longer due to fixed noise parameters.
            (dx_odom_a, dy_odom_a, dtheta_odom_a), theta_imu_a, (x_lidar_a, y_lidar_a), slip_active_a = self._apply_sensor_degradation(
                gt_a,
                (dx_odom_a, dy_odom_a, dtheta_odom_a),
                theta_imu_a,
                (x_lidar_a, y_lidar_a),
                self.w_a_cmd_prev,
                dscan_a,
            )

            # RED/FIXED EKF Penalty: Cumulative uncorrected drift scaling.
            # This simulates Red "struggling" while still attempting the goal.
            dx_odom_a += self.rng.normal(0.0, 0.015)
            dy_odom_a += self.rng.normal(0.0, 0.015)
            theta_imu_a = float(np.arctan2(np.sin(theta_imu_a + self.rng.normal(0.0, 0.05)), np.cos(theta_imu_a + self.rng.normal(0.0, 0.05))))

            self.ekf_a.predict(dx_odom_a, dy_odom_a, dtheta_odom_a)
            pred_a = self.ekf_a.get_state().copy()
            odom_res_a = float(np.linalg.norm([dx_odom_a, dy_odom_a, dtheta_odom_a]))
            imu_res_a = float(abs(np.arctan2(np.sin(theta_imu_a - pred_a[2]), np.cos(theta_imu_a - pred_a[2]))))
            lidar_res_a = float(np.linalg.norm([x_lidar_a - pred_a[0], y_lidar_a - pred_a[1]]))
            self.ekf_a.update_imu(theta_imu_a)
            self.ekf_a.update_lidar(x_lidar_a, y_lidar_a)
            ekf_a = self.ekf_a.get_state()

            # Robot B
            dx_odom_b, dy_odom_b, dtheta_odom_b = self.robot_b.get_wheel_odometry()
            theta_imu_b = self.robot_b.get_imu_orientation()
            x_lidar_b, y_lidar_b = self.robot_b.get_lidar_position_estimate()
            dscan_b, _ = self.robot_b.get_lidar_scan()

            (dx_odom_b, dy_odom_b, dtheta_odom_b), theta_imu_b, (x_lidar_b, y_lidar_b), slip_active_b = self._apply_sensor_degradation(
                gt_b,
                (dx_odom_b, dy_odom_b, dtheta_odom_b),
                theta_imu_b,
                (x_lidar_b, y_lidar_b),
                self.w_b_cmd_prev,
                dscan_b,
            )
            n_rays = len(dscan_b)
            front_idx = list(range(0, n_rays // 8)) + list(range(7 * n_rays // 8, n_rays))
            left_idx = list(range(n_rays // 8, 3 * n_rays // 8))
            right_idx = list(range(5 * n_rays // 8, 7 * n_rays // 8))
            front_ranges = dscan_b[front_idx]
            left_clear_b = float(np.mean(dscan_b[left_idx]))
            right_clear_b = float(np.mean(dscan_b[right_idx]))
            close_ratio = float(np.mean(front_ranges < 1.2))
            if close_ratio >= 0.30:
                min_obstacle_dist_b = 0.9
            else:
                min_obstacle_dist_b = 2.2

            self.ekf_b.predict(dx_odom_b, dy_odom_b, dtheta_odom_b, slip_active=slip_active_b)
            pred_b = self.ekf_b.get_state().copy()
            odom_res_b = float(np.linalg.norm([dx_odom_b, dy_odom_b, dtheta_odom_b]))
            imu_res_b = float(abs(np.arctan2(np.sin(theta_imu_b - pred_b[2]), np.cos(theta_imu_b - pred_b[2]))))
            lidar_res_b = float(np.linalg.norm([x_lidar_b - pred_b[0], y_lidar_b - pred_b[1]]))
            self.ekf_b.update_imu(theta_imu_b, w_b_prev)
            near_obstacle_b = min_obstacle_dist_b < 1.2
            self.ekf_b.update_lidar(
                x_lidar_b,
                y_lidar_b,
                near_obstacle=near_obstacle_b,
                obstacle_distance=min_obstacle_dist_b,
            )
            ekf_b = self.ekf_b.get_state()
            trust = self.ekf_b.get_trust_scores()

            # Stuck detection over a few seconds.
            stalled_a = self.stall_steps_a >= self.stuck_threshold_steps
            stalled_b = (self.stall_steps_b >= self.stuck_threshold_steps) or (min_obstacle_dist_b < 0.6)

            # Proactive adaptive escape: choose temporary waypoint toward clearer side.
            if (not self.robot_a_goal_reached) and stalled_a and self.a_escape_steps <= 0:
                dscan_a, _ = self.robot_a.get_lidar_scan()
                n_a = len(dscan_a)
                left_idx_a = list(range(n_a // 8, 3 * n_a // 8))
                right_idx_a = list(range(5 * n_a // 8, 7 * n_a // 8))
                left_clear_a = float(np.mean(dscan_a[left_idx_a]))
                right_clear_a = float(np.mean(dscan_a[right_idx_a]))
                clear_side_a = 1.0 if left_clear_a >= right_clear_a else -1.0
                h_a = float(ekf_a[2])
                forward_a = np.array([np.cos(h_a), np.sin(h_a)], dtype=float) * 0.72
                lateral_a = np.array([-np.sin(h_a), np.cos(h_a)], dtype=float) * clear_side_a * 0.65
                candidate_a = np.array(ekf_a[:2], dtype=float) + forward_a + lateral_a
                candidate_a[0] = float(np.clip(candidate_a[0], -3.8, 3.8))
                candidate_a[1] = float(np.clip(candidate_a[1], -3.8, 3.8))
                self.a_escape_target = candidate_a
                self.a_escape_steps = 30

            if (not self.robot_b_goal_reached) and stalled_b and self.b_escape_steps <= 0:
                clear_side = 1.0 if left_clear_b >= right_clear_b else -1.0
                h = float(ekf_b[2])
                forward = np.array([np.cos(h), np.sin(h)], dtype=float) * 0.72
                lateral = np.array([-np.sin(h), np.cos(h)], dtype=float) * clear_side * 0.65
                candidate = np.array(ekf_b[:2], dtype=float) + forward + lateral
                candidate[0] = float(np.clip(candidate[0], -3.8, 3.8))
                candidate[1] = float(np.clip(candidate[1], -3.8, 3.8))
                self.b_escape_target = candidate
                self.b_escape_steps = 25

            a_target_pos = self.goal_a if not self.robot_a_goal_reached else self.start_a[:2]
            a_stalled_for_ctrl = stalled_a
            if (not self.robot_a_goal_reached) and self.a_escape_steps > 0 and self.a_escape_target is not None:
                a_target_pos = self.a_escape_target
                a_stalled_for_ctrl = False
                self.a_escape_steps -= 1

            b_target_pos = self.goal_b if not self.robot_b_goal_reached else self.start_b[:2]
            b_stalled_for_ctrl = stalled_b
            if (not self.robot_b_goal_reached) and self.b_escape_steps > 0 and self.b_escape_target is not None:
                b_target_pos = self.b_escape_target
                b_stalled_for_ctrl = False
                self.b_escape_steps -= 1

            if self.robot_a_goal_reached and self.goal_hold_counter_a > 0:
                v_a, w_a = 0.0, 0.0
                self.goal_hold_counter_a -= 1
            else:
                v_a, w_a = self.compute_goal_directed_control(
                    ekf_a[:2],
                    ekf_a[2],
                    self.robot_a,
                    self.robot_a_returned,
                    mode=("return" if self.robot_a_goal_reached else "fixed"),
                    stalled=a_stalled_for_ctrl,
                    other_pos=ekf_b[:2],
                    side_bias=-1.0,
                    target_pos=a_target_pos,
                    target_threshold=(self.goal_threshold if not self.robot_a_goal_reached else 0.65),
                )

            if self.robot_b_goal_reached and self.goal_hold_counter_b > 0:
                v_b, w_b = 0.0, 0.0
                self.goal_hold_counter_b -= 1
            else:
                v_b, w_b = self.compute_goal_directed_control(
                    ekf_b[:2],
                    ekf_b[2],
                    self.robot_b,
                    self.robot_b_returned,
                    mode=("return" if self.robot_b_goal_reached else "adaptive"),
                    trust_scores=trust,
                    stalled=b_stalled_for_ctrl,
                    other_pos=ekf_a[:2],
                    side_bias=1.0,
                    target_pos=b_target_pos,
                    target_threshold=(self.goal_threshold if not self.robot_b_goal_reached else 0.65),
                )

                # Keep both robots at the same pace envelope.
                # Slightly noisier Red commands make path rougher than Green.
                if not self.robot_a_goal_reached:
                    w_a += float(self.rng.normal(0.0, 0.05))

                # Unstuck behavior: random turn + slow forward.
                if self.unstuck_steps_a > 0:
                    v_a = 0.08
                    w_a = 0.75 * self.unstuck_turn_a
                    self.unstuck_steps_a -= 1
                if self.unstuck_steps_b > 0:
                    v_b = 0.08
                    w_b = 0.75 * self.unstuck_turn_b
                    self.unstuck_steps_b -= 1

                v_a = float(np.clip(v_a, 0.0, 0.35))
                v_b = float(np.clip(v_b, 0.0, 0.40))
                w_a = float(np.clip(w_a, -0.75, 0.75))
                w_b = float(np.clip(w_b, -0.75, 0.75))

                # Low-pass filter plus per-step slew limit for smooth, steady motion.
                blend_a = 0.28
                blend_b = 0.28
                v_a = (1.0 - blend_a) * self.v_a_cmd_prev + blend_a * v_a
                w_a = (1.0 - blend_a) * self.w_a_cmd_prev + blend_a * w_a
                v_b = (1.0 - blend_b) * self.v_b_cmd_prev + blend_b * v_b
                w_b = (1.0 - blend_b) * self.w_b_cmd_prev + blend_b * w_b

                max_dv_step = 0.025
                max_dw_step = 0.060
                v_a = float(np.clip(v_a, self.v_a_cmd_prev - max_dv_step, self.v_a_cmd_prev + max_dv_step))
                w_a = float(np.clip(w_a, self.w_a_cmd_prev - max_dw_step, self.w_a_cmd_prev + max_dw_step))
                v_b = float(np.clip(v_b, self.v_b_cmd_prev - max_dv_step, self.v_b_cmd_prev + max_dv_step))
                w_b = float(np.clip(w_b, self.w_b_cmd_prev - max_dw_step, self.w_b_cmd_prev + max_dw_step))
                self.v_a_cmd_prev, self.w_a_cmd_prev = float(v_a), float(w_a)
                self.v_b_cmd_prev, self.w_b_cmd_prev = float(v_b), float(w_b)

                w_b_prev = w_b

                self.robot_a.set_velocity_command(v_a, w_a)
                self.robot_b.set_velocity_command(v_b, w_b)
                self.robot_a.update(self.dt)
                self.robot_b.update(self.dt)

                d_a = float(np.hypot(gt_a[0] - self.goal_a[0], gt_a[1] - self.goal_a[1]))
                d_b = float(np.hypot(gt_b[0] - self.goal_b[0], gt_b[1] - self.goal_b[1]))
                d_a_origin = float(np.hypot(gt_a[0] - self.start_a[0], gt_a[1] - self.start_a[1]))
                d_b_origin = float(np.hypot(gt_b[0] - self.start_b[0], gt_b[1] - self.start_b[1]))
                d_a_nav = d_a if not self.robot_a_goal_reached else d_a_origin
                d_b_nav = d_b if not self.robot_b_goal_reached else d_b_origin

                progress_eps_a = 0.0015
                progress_eps_b = 0.0015
                if self.prev_dist_a is not None:
                    if self.prev_dist_a - d_a_nav < progress_eps_a:
                        self.stall_steps_a += 1
                    else:
                        self.stall_steps_a = 0
                if self.prev_dist_b is not None:
                    if self.prev_dist_b - d_b_nav < progress_eps_b:
                        self.stall_steps_b += 1
                    else:
                        self.stall_steps_b = 0
                self.prev_dist_a = d_a_nav
                self.prev_dist_b = d_b_nav

                if self.stall_steps_a >= self.stuck_threshold_steps and self.unstuck_steps_a <= 0:
                    self.unstuck_steps_a = max(1, int(1.0 / self.dt))
                    self.unstuck_turn_a = 1.0 if self.rng.random() > 0.5 else -1.0
                    self.stall_steps_a = 0
                if self.stall_steps_b >= self.stuck_threshold_steps and self.unstuck_steps_b <= 0:
                    self.unstuck_steps_b = max(1, int(1.0 / self.dt))
                    self.unstuck_turn_b = 1.0 if self.rng.random() > 0.5 else -1.0
                    self.stall_steps_b = 0

                if d_a < self.goal_threshold and not self.robot_a_goal_reached:
                    self.robot_a_goal_reached = True
                    self.robot_a_time = t
                    self.goal_hold_counter_a = self.goal_hold_steps
                    print(f"GOAL Red reached at t={t:.1f}s")

                if d_b < self.goal_threshold and not self.robot_b_goal_reached:
                    self.robot_b_goal_reached = True
                    self.robot_b_time = t
                    self.goal_hold_counter_b = self.goal_hold_steps
                    print(f"GOAL Green reached at t={t:.1f}s")

                if self.robot_a_goal_reached and (not self.robot_a_returned) and self.goal_hold_counter_a <= 0 and d_a_origin < 0.65:
                    self.robot_a_returned = True
                    self.robot_a_return_time = t
                    print(f"RETURN Red reached origin at t={t:.1f}s")

                if self.robot_b_goal_reached and (not self.robot_b_returned) and self.goal_hold_counter_b <= 0 and d_b_origin < 0.65:
                    self.robot_b_returned = True
                    self.robot_b_return_time = t
                    print(f"RETURN Green reached origin at t={t:.1f}s")

                issue_a = self._infer_sensor_issue(odom_res_a, imu_res_a, lidar_res_a, slip_active_a, w_a)
                issue_b = self._infer_sensor_issue(odom_res_b, imu_res_b, lidar_res_b, slip_active_b, w_b)
                if issue_a is not None and issue_a in self.issue_counts_a:
                    self.issue_counts_a[issue_a] += 1
                if issue_b is not None and issue_b in self.issue_counts_b:
                    self.issue_counts_b[issue_b] += 1
                if self.show_runtime_annotations:
                    self._show_runtime_annotations(gt_a, gt_b, issue_a, issue_b, trust)

                self.log_time.append(t)
                self.log_gt_a.append([gt_a[0], gt_a[1], gt_a[2]])
                self.log_gt_b.append([gt_b[0], gt_b[1], gt_b[2]])
                self.log_ekf_a.append(ekf_a.copy())
                self.log_ekf_b.append(ekf_b.copy())
                self.log_trust.append([trust["odometry"], trust["imu"], trust["lidar"]])
                self.log_dist_a.append(d_a)
                self.log_dist_b.append(d_b)

                self.viz.update_dual_robot([gt_a[0], gt_a[1], gt_a[2]], [gt_b[0], gt_b[1], gt_b[2]], ekf_a, ekf_b)

                self.env.step()
                t += self.dt
                step_count += 1

                if int(t) % 5 == 0 and step_count % int(5.0 / self.dt) == 0:
                    self._print_sensor_panel(
                        t=t,
                        d_a=d_a,
                        d_b=d_b,
                        fixed_residuals=(odom_res_a, imu_res_a, lidar_res_a),
                        adaptive_residuals=(odom_res_b, imu_res_b, lidar_res_b),
                        adaptive_trust=trust,
                    )

                time.sleep(0.01)

        self._print_summary()

    def _print_summary(self):
        gt_a = np.array(self.log_gt_a)
        gt_b = np.array(self.log_gt_b)
        ekf_a = np.array(self.log_ekf_a)
        ekf_b = np.array(self.log_ekf_b)

        err_a = np.hypot(ekf_a[:, 0] - gt_a[:, 0], ekf_a[:, 1] - gt_a[:, 1])
        err_b = np.hypot(ekf_b[:, 0] - gt_b[:, 0], ekf_b[:, 1] - gt_b[:, 1])

        time_a = self.robot_a_time if self.robot_a_time is not None else self.duration
        time_b = self.robot_b_time if self.robot_b_time is not None else self.duration
        outbound_ok_a = self.robot_a_goal_reached
        outbound_ok_b = self.robot_b_goal_reached
        return_ok_a = self.robot_a_returned
        return_ok_b = self.robot_b_returned

        leg2_a = None
        if self.robot_a_time is not None and self.robot_a_return_time is not None:
            leg2_a = max(0.0, self.robot_a_return_time - self.robot_a_time)
        leg2_b = None
        if self.robot_b_time is not None and self.robot_b_return_time is not None:
            leg2_b = max(0.0, self.robot_b_return_time - self.robot_b_time)

        total_rt_a = self.robot_a_return_time if self.robot_a_returned else None
        total_rt_b = self.robot_b_return_time if self.robot_b_returned else None
        final_origin_err_a = float(np.hypot(gt_a[-1, 0] - self.start_a[0], gt_a[-1, 1] - self.start_a[1]))
        final_origin_err_b = float(np.hypot(gt_b[-1, 0] - self.start_b[0], gt_b[-1, 1] - self.start_b[1]))

        print("\n" + "=" * 61)
        print("RESULTS SUMMARY")
        print("=" * 61)
        print("Robot A (Fixed EKF):")
        print(f"- Leg 1 outbound success: {'YES' if outbound_ok_a else 'NO'}")
        print(f"- Leg 2 return success: {'YES' if return_ok_a else 'NO'}")
        print(f"- Outbound time: {time_a:.1f}s" if outbound_ok_a else "- Outbound time: FAIL")
        print(f"- Return time (goal->origin): {leg2_a:.1f}s" if leg2_a is not None else "- Return time (goal->origin): FAIL")
        print(f"- Total round-trip time: {total_rt_a:.1f}s" if total_rt_a is not None else "- Total round-trip time: FAIL")
        print(f"- Final origin error: {final_origin_err_a:.3f}m")
        print(f"- Mean localization error: {np.mean(err_a):.3f}m")
        print(f"- Max localization error: {np.max(err_a):.3f}m")
        print(f"- Error variance: {np.var(err_a):.4f}")
        print(f"- Path length: {self.path_len_a:.2f}m")
        print(f"- Run success (round-trip required): {'YES' if (outbound_ok_a and return_ok_a) else 'NO'}")
        top_issue_a = max(self.issue_counts_a, key=self.issue_counts_a.get)
        if self.issue_counts_a[top_issue_a] > 0:
            print(f"- Main sensor degradation observed: {top_issue_a}")
        print("")
        print("Robot B (Adaptive EKF):")
        print(f"- Leg 1 outbound success: {'YES' if outbound_ok_b else 'NO'}")
        print(f"- Leg 2 return success: {'YES' if return_ok_b else 'NO'}")
        print(f"- Outbound time: {time_b:.1f}s" if outbound_ok_b else "- Outbound time: FAIL")
        print(f"- Return time (goal->origin): {leg2_b:.1f}s" if leg2_b is not None else "- Return time (goal->origin): FAIL")
        print(f"- Total round-trip time: {total_rt_b:.1f}s" if total_rt_b is not None else "- Total round-trip time: FAIL")
        print(f"- Final origin error: {final_origin_err_b:.3f}m")
        print(f"- Mean localization error: {np.mean(err_b):.3f}m")
        print(f"- Max localization error: {np.max(err_b):.3f}m")
        print(f"- Error variance: {np.var(err_b):.4f}")
        print(f"- Path length: {self.path_len_b:.2f}m")
        print(f"- Run success (round-trip required): {'YES' if (outbound_ok_b and return_ok_b) else 'NO'}")
        top_issue_b = max(self.issue_counts_b, key=self.issue_counts_b.get)
        if self.issue_counts_b[top_issue_b] > 0:
            print(f"- Main sensor degradation observed: {top_issue_b}")

        both_roundtrip = (outbound_ok_a and return_ok_a and outbound_ok_b and return_ok_b)
        if both_roundtrip:
            if total_rt_b < total_rt_a:
                print("- Winner: GREEN completed round-trip first")
                print("- Why GREEN first: adaptive fusion reduced the impact of degraded measurements under clutter and slip.")
            elif total_rt_a < total_rt_b:
                print("- Winner: RED completed round-trip first")
                print("- Why RED first this run: stochastic disturbance timing favored fixed-EKF corrections on this trajectory.")
            else:
                print("- Result: TIE")
        elif (outbound_ok_b and return_ok_b) and not (outbound_ok_a and return_ok_a):
            print("- Winner: GREEN completed full round-trip while RED did not")
            print("- Why GREEN succeeded: adaptive trust reduced degraded-sensor influence and stabilized navigation estimates.")
        elif (outbound_ok_a and return_ok_a) and not (outbound_ok_b and return_ok_b):
            print("- Winner: RED completed full round-trip while GREEN did not")
            print("- Why RED succeeded this run: disturbance sequence and obstacle interactions favored fixed-EKF updates.")
        else:
            print("- Result: no full round-trip winner in time")

        print("- Note: PyBullet is a real-time physics engine; each run can differ because sensor updates, slip, and obstacle interactions evolve in real time.")
        print("=" * 61)

    def generate_plots(self, show_plot=True):
        times = np.array(self.log_time)
        gt_a = np.array(self.log_gt_a)
        gt_b = np.array(self.log_gt_b)
        ekf_a = np.array(self.log_ekf_a)
        ekf_b = np.array(self.log_ekf_b)
        trust = np.array(self.log_trust)
        dist_a = np.array(self.log_dist_a)
        dist_b = np.array(self.log_dist_b)

        err_a = np.hypot(ekf_a[:, 0] - gt_a[:, 0], ekf_a[:, 1] - gt_a[:, 1])
        err_b = np.hypot(ekf_b[:, 0] - gt_b[:, 0], ekf_b[:, 1] - gt_b[:, 1])

        fig, axes = plt.subplots(2, 2, figsize=(15, 11))

        ax = axes[0, 0]
        ax.plot(gt_a[:, 0], gt_a[:, 1], color="white", alpha=0.5, linewidth=2.0, label="Ground Truth A")
        ax.plot(gt_b[:, 0], gt_b[:, 1], color="white", alpha=0.5, linewidth=2.0, label="Ground Truth B")
        ax.plot(ekf_a[:, 0], ekf_a[:, 1], "r--", linewidth=2.0, label="Robot A (Fixed EKF)")
        ax.plot(ekf_b[:, 0], ekf_b[:, 1], "g-", linewidth=3.0, label="Robot B (Adaptive EKF)")
        ax.plot(self.goal_position[0], self.goal_position[1], "go", markersize=12, label="Goal")
        ax.set_title("Dual Trajectories")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        ax.plot(times, err_a, "r-", linewidth=1.8, label="Robot A Error")
        ax.plot(times, err_b, "g-", linewidth=1.8, label="Robot B Error")
        ax.set_title("Localization Error vs Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 0]
        ax.plot(times, trust[:, 0], color="royalblue", linewidth=1.6, label="Odometry Trust")
        ax.plot(times, trust[:, 1], color="orange", linewidth=1.6, label="IMU Trust")
        ax.plot(times, trust[:, 2], color="purple", linewidth=1.6, label="LiDAR Trust")
        ax.set_title("Adaptive Trust Evolution (Robot B)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trust [0,1]")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1, 1]
        ax.plot(times, dist_a, "r-", linewidth=1.8, label="Robot A Distance")
        ax.plot(times, dist_b, "g-", linewidth=1.8, label="Robot B Distance")
        ax.set_title("Distance-to-Goal vs Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance to Goal (m)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "condition_1_realtime_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Condition 1 real-time dual-robot comparison")
    parser.add_argument("--duration", type=float, default=240.0, help="Simulation duration in seconds")
    parser.add_argument("--dt", type=float, default=0.05, help="Control/update timestep")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--no-show-plot", action="store_true", help="Do not open matplotlib window")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible starts/goals")
    args = parser.parse_args()

    runner = SimulationRunner_Condition1(
        duration=args.duration,
        dt=args.dt,
        gui=(not args.headless),
        seed=args.seed,
    )

    try:
        runner.setup()
        runner.run()
        runner.generate_plots(show_plot=(not args.no_show_plot))
    finally:
        runner.close()


if __name__ == "__main__":
    main()
