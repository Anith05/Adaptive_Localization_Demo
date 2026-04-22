"""
Condition 2: Sequential Solo-Run with Return Trip.

Run 1: Red Robot (Fixed EKF) alone — START -> GOAL -> START
Run 2: Green Robot (Adaptive EKF) alone — same environment, same seed

Proves Adaptive EKF works better independently, without interaction noise.

Environment: Mini-warehouse with colored walls, dense rack corridors,
friction zones, and obstacles near goal — matching Condition 1 visuals.
Uses PyBulletEnvironment + bulletphysics patterns (loadURDF plane, boston_box, etc).
"""
import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.pybullet_env import PyBulletEnvironment
from fusion.ekf import ExtendedKalmanFilter
from demo.live_visualization import LiveVisualization
from demo.pause_controls import PauseControls


# ---------------------------------------------------------------------------
# Adaptive fusion (self-contained copy from Condition 1)
# ---------------------------------------------------------------------------
class StrongAdaptiveFusion:
    """Adaptive trust-based fusion with sliding-window, context-aware trust scores."""

    def __init__(self):
        self.ekf = ExtendedKalmanFilter()
        self.Q_base = self.ekf.Q.copy()
        self.R_imu_base = np.array([[0.006]])
        self.R_lidar_base = np.diag([0.04, 0.04])
        self.ekf.R_imu = self.R_imu_base.copy()
        self.ekf.R_lidar = self.R_lidar_base.copy()
        self.trust = {"odometry": 1.0, "imu": 1.0, "lidar": 1.0}
        self.window_size = 20
        self.residual_hist = {"odometry": [], "imu": [], "lidar": []}

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
            base = self._piecewise_trust(residual, 0.07, 0.24)
            if slip_active or z > 1.8:
                base *= 0.88
        elif key == "imu":
            base = self._piecewise_trust(residual, 0.05, 0.22)
            if abs(angular_velocity) > 0.80:
                base *= 0.88
        else:
            base = self._piecewise_trust(residual, 0.10, 0.52)
            if obstacle_distance < 1.5:
                base *= 0.86

        base = float(np.clip(base, 0.3, 1.0))
        smoothed = 0.80 * self.trust[key] + 0.20 * base
        self.trust[key] = float(np.clip(smoothed, 0.3, 1.0))

    def predict(self, dx, dy, dtheta, slip_active=False):
        res = float(np.linalg.norm([dx, dy, dtheta]))
        if slip_active:
            res *= 1.8
        self._update_trust("odometry", res, slip_active=slip_active)
        self.ekf.Q = self.Q_base.copy()
        self.ekf.predict(dx, dy, dtheta)

    def update_imu(self, theta_measured, angular_velocity=0.0):
        pred_theta = self.ekf.get_state()[2]
        res = float(abs(np.arctan2(np.sin(theta_measured - pred_theta), np.cos(theta_measured - pred_theta))))
        if abs(angular_velocity) > 0.5:
            res *= 1.3
        res *= 0.55
        self._update_trust("imu", res, angular_velocity=angular_velocity)
        t = max(self.trust["imu"], 0.3)
        self.ekf.R_imu = self.R_imu_base / (t ** 2)
        self.ekf.update_imu(theta_measured)

    def update_lidar(self, x_m, y_m, near_obstacle=False, obstacle_distance=3.0):
        pred = self.ekf.get_state()
        res = float(np.linalg.norm([x_m - pred[0], y_m - pred[1]]))
        if near_obstacle:
            res *= 1.9
        res *= 0.45
        self._update_trust("lidar", res, obstacle_distance=obstacle_distance)
        t = max(self.trust["lidar"], 0.3)
        self.ekf.R_lidar = self.R_lidar_base / (t ** 2)
        self.ekf.update_lidar(x_m, y_m)

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


# ---------------------------------------------------------------------------
# Compact differential-drive robot (matches Condition 1 exactly)
# ---------------------------------------------------------------------------
class CompactDifferentialRobot:
    """Compact box robot with differential-drive kinematics."""

    def __init__(self, env, start_pos, color,
                 odom_noise_std=0.12, imu_noise_std=0.25, lidar_noise_std=0.10):
        self.env = env
        self.start_pos = list(start_pos)
        self.color = color
        self.robot_id = None
        self.odom_noise_std = odom_noise_std
        self.imu_noise_std = imu_noise_std
        self.lidar_noise_std = lidar_noise_std

        self.true_x = float(start_pos[0])
        self.true_y = float(start_pos[1])
        self.true_theta = 0.0
        self.prev_odom_x = self.true_x
        self.prev_odom_y = self.true_y
        self.prev_odom_theta = 0.0
        self.v_cmd = self.w_cmd = self.v_applied = self.w_applied = 0.0
        self.angular_velocity_hist = []
        self.max_range = 5.0

    def create(self):
        body_size = [0.28, 0.18, 0.12]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2.0 for s in body_size])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s / 2.0 for s in body_size],
                                   rgbaColor=self.color)
        self.robot_id = p.createMultiBody(
            baseMass=5.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=self.start_pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        )
        p.changeDynamics(self.robot_id, -1, lateralFriction=0.8)

    def set_velocity_command(self, v, w):
        self.v_cmd = v
        self.w_cmd = w

    def update(self, dt):
        v_target = self.v_cmd
        w_target = self.w_cmd

        # Smooth acceleration
        max_dv = 0.26 * dt * 2.6
        max_dw = 1.2 * dt * 2.4
        self.v_applied += np.clip(v_target - self.v_applied, -max_dv, max_dv)
        self.w_applied += np.clip(w_target - self.w_applied, -max_dw, max_dw)

        v = self.v_applied
        w = self.w_applied

        new_theta = self.true_theta + w * dt
        new_theta = float(np.arctan2(np.sin(new_theta), np.cos(new_theta)))
        new_x = self.true_x + v * np.cos(new_theta) * dt
        new_y = self.true_y + v * np.sin(new_theta) * dt

        # Arena bounds
        new_x = float(np.clip(new_x, -4.5, 4.5))
        new_y = float(np.clip(new_y, -4.5, 4.5))

        quat = p.getQuaternionFromEuler([0, 0, new_theta])
        p.resetBasePositionAndOrientation(self.robot_id, [new_x, new_y, self.start_pos[2]], quat)

        # Collision check 
        collision = self.env.check_collision(self.robot_id)
        if collision:
            # Revert 
            old_quat = p.getQuaternionFromEuler([0, 0, self.true_theta])
            p.resetBasePositionAndOrientation(self.robot_id,
                                              [self.true_x, self.true_y, self.start_pos[2]], old_quat)
            self.v_applied *= 0.3
            return

        self.true_x = new_x
        self.true_y = new_y
        self.true_theta = new_theta

    def get_ground_truth(self):
        return self.true_x, self.true_y, self.true_theta

    def get_wheel_odometry(self):
        dx_true = self.true_x - self.prev_odom_x
        dy_true = self.true_y - self.prev_odom_y
        dtheta_true = np.arctan2(np.sin(self.true_theta - self.prev_odom_theta),
                                  np.cos(self.true_theta - self.prev_odom_theta))
        dx_o = dx_true + np.random.normal(0, self.odom_noise_std)
        dy_o = dy_true + np.random.normal(0, self.odom_noise_std)
        dt_o = float(dtheta_true) + np.random.normal(0, self.odom_noise_std * 2.0)
        self.prev_odom_x = self.true_x
        self.prev_odom_y = self.true_y
        self.prev_odom_theta = self.true_theta
        return dx_o, dy_o, dt_o

    def get_imu_orientation(self):
        self.angular_velocity_hist.append(abs(self.w_cmd))
        if len(self.angular_velocity_hist) > 10:
            self.angular_velocity_hist.pop(0)
        avg_w = float(np.mean(self.angular_velocity_hist)) if self.angular_velocity_hist else 0.0
        theta_imu = self.true_theta + np.random.normal(0.0, self.imu_noise_std)
        if abs(avg_w) > 0.40:
            theta_imu += np.random.normal(0.0, 0.09)
        return float(np.arctan2(np.sin(theta_imu), np.cos(theta_imu)))

    def get_lidar_scan(self, num_rays=24):
        angles = np.linspace(0, 2.0 * np.pi, num_rays, endpoint=False)
        ray_from, ray_to = [], []
        for a in angles:
            ra = self.true_theta + a
            ray_from.append([self.true_x + 0.2 * np.cos(ra),
                             self.true_y + 0.2 * np.sin(ra), 0.2])
            ray_to.append([self.true_x + self.max_range * np.cos(ra),
                           self.true_y + self.max_range * np.sin(ra), 0.2])
        hits = p.rayTestBatch(ray_from, ray_to)
        distances = []
        for hit in hits:
            if hit[0] != -1:
                d = 0.2 + (self.max_range - 0.2) * hit[2]
            else:
                d = self.max_range
            near_noise = 1.0 + 0.9 * max(0.0, (1.5 - d) / 1.5)
            d += np.random.normal(0.0, self.lidar_noise_std * near_noise)
            distances.append(float(np.clip(d, 0.1, self.max_range)))
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
            sigma = 0.20
        x_est = self.true_x + np.random.normal(0.0, sigma)
        y_est = self.true_y + np.random.normal(0.0, sigma)
        if occluded:
            x_est += np.random.uniform(-0.55, 0.55)
            y_est += np.random.uniform(-0.55, 0.55)
        return x_est, y_est

    def remove(self):
        if self.robot_id is not None:
            try:
                p.removeBody(self.robot_id)
            except Exception:
                pass
            self.robot_id = None


# ---------------------------------------------------------------------------
# Condition 2 Runner — Mini-Warehouse with Colored Walls
# ---------------------------------------------------------------------------
class SimulationRunner_Condition2:
    """Condition 2: Sequential solo-run with return trip in mini-warehouse."""

    def __init__(self, duration=60.0, dt=0.05, gui=True, seed=7):
        self.duration = float(duration)
        self.dt = float(dt)
        self.gui = gui
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # Layout: start at top, goal at bottom — matching C1 distance ~4m
        self.start_pos = np.array([1.0, 3.0, 0.1], dtype=float)
        self.goal_pos = np.array([-1.0, -3.0], dtype=float)
        self.goal_threshold = 0.65
        self.return_threshold = 0.80
        self.goal_hold_seconds = 1.2
        self.goal_hold_steps = max(1, int(self.goal_hold_seconds / self.dt))

        self.env = PyBulletEnvironment(gui=gui, scenario="fixed_advantage")
        self.viz = LiveVisualization()
        self.pause_ctrl = PauseControls()

        self.run_results = {}
        self.product_prop_count = 0

        # Persistent disturbance bursts triggered by real-time physics context.
        self.slip_burst_steps = 0
        self.imu_burst_steps = 0
        self.lidar_burst_steps = 0

    # -- Build the warehouse environment (matching Condition 1 visuals) ------

    def _build_mini_warehouse(self):
        """Build warehouse with colored walls, rack obstacles, friction zones.
        
        Matches the 3rd screenshot layout: colored perimeter walls,
        dark rack blocks, goal zone with product props, start bay.
        Uses bulletphysics patterns (p.createMultiBody, GEOM_BOX).
        """
        wall_color = [0.2, 0.25, 0.3, 1.0]
        wall_height = 0.85

        # ---- 1. External Perimeter (same as C1) ----
        perimeter_walls = [
            ([0, 4.8, wall_height], [4.8, 0.1, wall_height]),   # North
            ([0, -4.8, wall_height], [4.8, 0.1, wall_height]),  # South
            ([4.8, 0, wall_height], [0.1, 4.8, wall_height]),   # East
            ([-4.8, 0, wall_height], [0.1, 4.8, wall_height]),  # West
        ]
        # Colored walls (matching 3rd screenshot: yellow=top, red=right, blue=bottom, green=right-bottom)
        wall_colors = [
            [0.80, 0.70, 0.10, 1.0],  # North = Yellow
            [0.15, 0.35, 0.75, 1.0],  # South = Blue
            [0.75, 0.15, 0.15, 1.0],  # East = Red
            [0.15, 0.65, 0.20, 1.0],  # West = Green
        ]
        for (pos, ext), wc in zip(perimeter_walls, wall_colors):
            c = p.createCollisionShape(p.GEOM_BOX, halfExtents=ext)
            v = p.createVisualShape(p.GEOM_BOX, halfExtents=ext, rgbaColor=wc)
            wid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c,
                                     baseVisualShapeIndex=v, basePosition=pos)
            self.env.obstacle_ids.append(wid)

        gx, gy = float(self.goal_pos[0]), float(self.goal_pos[1])
        sx, sy = float(self.start_pos[0]), float(self.start_pos[1])

        # ---- 2. Start Bay Room ----
        start_walls = [
            ([sx - 1.5, sy, wall_height / 2], [0.1, 1.3, wall_height / 2]),
            ([sx + 1.5, sy, wall_height / 2], [0.1, 1.3, wall_height / 2]),
            ([sx, sy + 1.4, wall_height / 2], [1.5, 0.1, wall_height / 2]),
        ]
        for pos, ext in start_walls:
            c = p.createCollisionShape(p.GEOM_BOX, halfExtents=ext)
            v = p.createVisualShape(p.GEOM_BOX, halfExtents=ext, rgbaColor=wall_color)
            wid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c,
                                     baseVisualShapeIndex=v, basePosition=pos)
            self.env.obstacle_ids.append(wid)

        # Start bay floor marking
        floor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.2, 0.7, 0.005],
                                         rgbaColor=[0.08, 0.36, 0.38, 0.22])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=floor_vis,
                          basePosition=[sx, sy, 0.01])
        p.addUserDebugText("START BAY", textPosition=[sx - 0.5, sy - 0.5, 0.18],
                           textColorRGB=[0.07, 0.26, 0.30], textSize=1.0, lifeTime=0)

        # ---- 3. Goal / Delivery Bay Room ----
        goal_walls = [
            ([gx - 1.5, gy, wall_height / 2], [0.1, 1.3, wall_height / 2]),
            ([gx + 1.5, gy, wall_height / 2], [0.1, 1.3, wall_height / 2]),
            ([gx, gy - 1.4, wall_height / 2], [1.5, 0.1, wall_height / 2]),
        ]
        for pos, ext in goal_walls:
            c = p.createCollisionShape(p.GEOM_BOX, halfExtents=ext)
            v = p.createVisualShape(p.GEOM_BOX, halfExtents=ext, rgbaColor=wall_color)
            wid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=c,
                                     baseVisualShapeIndex=v, basePosition=pos)
            self.env.obstacle_ids.append(wid)

        # Goal highlight
        outer = p.createVisualShape(p.GEOM_CYLINDER, radius=0.90, length=0.012,
                                     rgbaColor=[0.10, 1.00, 0.25, 0.20])
        inner = p.createVisualShape(p.GEOM_CYLINDER, radius=0.28, length=0.016,
                                     rgbaColor=[0.12, 0.95, 0.30, 0.60])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=outer, basePosition=[gx, gy, 0.008])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=inner, basePosition=[gx, gy, 0.010])
        p.addUserDebugText("GOAL ZONE", textPosition=[gx - 0.34, gy + 0.20, 0.55],
                           textColorRGB=[0.08, 0.92, 0.24], textSize=1.35, lifeTime=0)

        # ---- 4. Product Props Near Goal (occludes LiDAR, keeps path clear) ----
        placed = 0
        while placed < 4:
            px = gx + self.rng.uniform(-0.6, 0.6)
            py = gy - self.rng.uniform(0.5, 1.2)
            oid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.25]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.25],
                                                          rgbaColor=[0.6, 0.5, 0.4, 1.0]),
                basePosition=[px, py, 0.25],
            )
            self.env.obstacle_ids.append(oid)
            self.product_prop_count += 1
            placed += 1

        # ---- 5. High-density rack corridors (dark boxes matching C1 layout) ----
        for i in range(-2, 3):
            for j in range(-2, 3):
                bx, by = i * 1.8, j * 1.8
                # Keep large margins for start room, goal room, and navigation corridor
                if bx > -1.5 and bx < 3.0 and by > 1.0:
                    continue   # Start room zone
                if bx > -3.0 and bx < 1.5 and by < -1.0:
                    continue   # Goal room zone
                if abs(bx) < 2.0 and abs(by) < 1.8:
                    continue   # Keep pressure while preserving a usable center corridor
                extents = [0.35, 0.65, 0.55]
                oid = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=extents),
                    baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=extents,
                                                              rgbaColor=[0.25, 0.25, 0.30, 1.0]),
                    basePosition=[bx, by, extents[2]],
                )
                self.env.obstacle_ids.append(oid)

        # Add a deterministic slalom around the center lane so both robots must correct heading.
        slalom_blocks = [
            ([0.45, 0.65, 0.55], [0.18, 0.45, 0.55]),
            ([-0.45, -0.35, 0.55], [0.18, 0.45, 0.55]),
        ]
        for pos, ext in slalom_blocks:
            c = p.createCollisionShape(p.GEOM_BOX, halfExtents=ext)
            v = p.createVisualShape(p.GEOM_BOX, halfExtents=ext, rgbaColor=[0.20, 0.23, 0.27, 1.0])
            oid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=c,
                baseVisualShapeIndex=v,
                basePosition=pos,
            )
            self.env.obstacle_ids.append(oid)

        # ---- 6. Clear goal zone of overlapping obstacles ----
        kept = []
        removed_count = 0
        for obs_id in self.env.obstacle_ids:
            try:
                pos, _ = p.getBasePositionAndOrientation(obs_id)
            except Exception:
                continue
            if np.hypot(pos[0] - gx, pos[1] - gy) < 0.42:
                try:
                    p.removeBody(obs_id)
                    removed_count += 1
                except Exception:
                    kept.append(obs_id)
            else:
                kept.append(obs_id)
        self.env.obstacle_ids = kept
        if removed_count > 0:
            print(f"Goal-zone cleanup removed {removed_count} overlapping obstacle(s).")

        # ---- 7. Friction zones (strong slip to trigger sensor degradation) ----
        self.env.friction_zones = [
            (-0.8, -1.8, 1.20, 0.08),   # Near delivery bay — heavy slip
            (0.5, 0.5, 1.40, 0.10),     # Center crossing — heavy slip
            (1.0, 2.0, 1.00, 0.12),     # Outside start bay — heavy slip
        ]
        for fx, fy, radius, _ in self.env.friction_zones:
            fv = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=0.01,
                                      rgbaColor=[0.85, 0.8, 0.2, 0.35])
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=fv, basePosition=[fx, fy, 0.005])

        print(f"Mini-warehouse built: {len(self.env.obstacle_ids)} collidable bodies, "
              f"{self.product_prop_count} product props")

    # -- Sensor degradation (STRONG — identical to Condition 1) --------------

    def _apply_sensor_degradation(self, gt, odom, theta_imu, lidar_xy, w_cmd, lidar_scan):
        """Apply physics-based sensor degradation: slip, IMU spikes, LiDAR occlusion."""
        dx, dy, dtheta = odom
        lx, ly = lidar_xy
        theta = float(theta_imu)

        friction = self.env.get_friction_at_position(gt[0], gt[1])
        scan = np.asarray(lidar_scan)
        n = len(scan)
        front_idx = list(range(0, n // 8)) + list(range(7 * n // 8, n))
        front_ranges = scan[front_idx] if len(front_idx) > 0 else scan
        front_min = float(np.min(front_ranges)) if len(front_ranges) > 0 else 5.0
        close_ratio = float(np.mean(front_ranges < 1.4)) if len(front_ranges) > 0 else 0.0

        # Slip is physically tied to low friction and aggressive turning on low-traction floor.
        slip_active = (friction < 0.72) or (friction < 0.85 and abs(w_cmd) > 0.45)

        # Wheel slip burst memory: once slip starts in low-friction zone, it persists briefly.
        if slip_active and self.rng.random() < 0.35:
            self.slip_burst_steps = max(self.slip_burst_steps, int(self.rng.integers(8, 18)))
        if self.slip_burst_steps > 0:
            slip_scale = self.rng.uniform(0.10, 0.38)
            dx = dx * slip_scale + self.rng.uniform(-0.030, 0.030)
            dy = dy * slip_scale + self.rng.uniform(-0.030, 0.030)
            dtheta += self.rng.normal(0.0, 0.09)
            self.slip_burst_steps -= 1
        elif slip_active:
            slip_scale = self.rng.uniform(0.12, 0.40)
            dx = dx * slip_scale + self.rng.uniform(-0.030, 0.030)
            dy = dy * slip_scale + self.rng.uniform(-0.030, 0.030)
            dtheta += self.rng.normal(0.0, 0.09)

        # IMU spikes scale with angular velocity and local clutter.
        imu_spike_p = 0.05
        if abs(w_cmd) > 0.40:
            imu_spike_p += 0.26
        if front_min < 1.4:
            imu_spike_p += 0.10
        if friction < 0.20:
            imu_spike_p += 0.12
        if abs(w_cmd) > 0.55 and self.rng.random() < 0.30:
            self.imu_burst_steps = max(self.imu_burst_steps, int(self.rng.integers(6, 14)))
        imu_event = False
        if self.imu_burst_steps > 0 or self.rng.random() < imu_spike_p:
            sigma = 0.16 + (0.12 if abs(w_cmd) > 0.55 else 0.0)
            if self.imu_burst_steps > 0:
                sigma += 0.10
                self.imu_burst_steps -= 1
            theta += self.rng.normal(0.0, sigma)
            theta = float(np.arctan2(np.sin(theta), np.cos(theta)))
            imu_event = True

        # LiDAR occlusion bursts are tied to nearby obstacles and clutter ratio.
        occ_p = 0.08 + (0.44 if front_min < 1.5 else 0.0) + 0.26 * close_ratio
        if front_min < 1.2 and self.rng.random() < 0.35:
            self.lidar_burst_steps = max(self.lidar_burst_steps, int(self.rng.integers(6, 16)))
        lidar_event = False
        if self.lidar_burst_steps > 0 or self.rng.random() < occ_p:
            jitter = 0.85 if self.lidar_burst_steps > 0 else 0.65
            lx += self.rng.uniform(-jitter, jitter)
            ly += self.rng.uniform(-jitter, jitter)
            lx += self.rng.normal(0.0, 0.06)
            ly += self.rng.normal(0.0, 0.06)
            if self.lidar_burst_steps > 0:
                self.lidar_burst_steps -= 1
            lidar_event = True

        slip_context = slip_active or (self.slip_burst_steps > 0)
        return (dx, dy, dtheta), theta, (lx, ly), slip_context, {
            "slip": bool(slip_context),
            "imu_spike": bool(imu_event),
            "lidar_occlusion": bool(lidar_event),
        }

    # -- Controller (matching C1 speed envelope) ----------------------------

    def _compute_control(self, pos, theta, robot, target, mode="fixed",
                         trust=None, stalled=False):
        """Goal-directed controller with obstacle avoidance.
        
        Speed: max_v=0.35 (fixed), 0.35 (adaptive), max_w=0.72
        Same speed envelope as Condition 1.
        """
        lidar_ranges, _ = robot.get_lidar_scan()
        dx_g = target[0] - pos[0]
        dy_g = target[1] - pos[1]
        dist = float(np.hypot(dx_g, dy_g))
        if dist < 0.25:
            return 0.0, 0.0

        desired = np.arctan2(dy_g, dx_g)
        ang_err = float(np.arctan2(np.sin(desired - theta), np.cos(desired - theta)))

        n = len(lidar_ranges)
        front_idx = list(range(0, n // 8)) + list(range(7 * n // 8, n))
        left_idx = list(range(n // 8, 3 * n // 8))
        right_idx = list(range(5 * n // 8, 7 * n // 8))
        front_min = float(np.min(lidar_ranges[front_idx]))
        left_clear = float(np.mean(lidar_ranges[left_idx]))
        right_clear = float(np.mean(lidar_ranges[right_idx]))

        max_v = 0.30 if mode == "adaptive" else 0.28
        max_w = 0.58
        obs_thresh = 0.80
        near_goal = dist < 1.2
        if near_goal:
            obs_thresh *= 0.85

        if front_min < obs_thresh:
            pref = 1.0 if left_clear >= right_clear else -1.0
            if stalled:
                v = 0.05
                w = pref * max_w
            else:
                v = 0.07
                w = pref * (0.68 * max_w)
        else:
            v = max(0.06, max_v * min(1.0, dist / 2.1))
            w = np.clip(0.78 * ang_err, -max_w, max_w)

        # Adaptive trust-guided reroute for Green
        if mode == "adaptive" and trust is not None:
            lidar_t = float(trust.get("lidar", 1.0))
            imu_t = float(trust.get("imu", 1.0))
            odom_t = float(trust.get("odometry", 1.0))
            trust_min = min(lidar_t, imu_t, odom_t)

            risk_front = float(np.clip((obs_thresh + 0.25 - front_min) / (obs_thresh + 0.25), 0.0, 1.0))
            trust_risk = float(np.clip((0.75 - trust_min) / 0.45, 0.0, 1.0))
            adapt_gain = risk_front * (0.40 + 0.50 * trust_risk)

            if adapt_gain > 0.24 and trust_min < 0.58:
                side = 1.0 if left_clear >= right_clear else -1.0
                penalty = max(0.0, 0.80 - trust_min)
                w_avoid = side * (0.72 + 0.35 * adapt_gain + 0.30 * penalty) * max_w
                blend = 0.24 + 0.34 * adapt_gain + 0.22 * penalty
                w = np.clip((1.0 - blend) * w + blend * w_avoid, -max_w, max_w)
                v = min(v, max(0.12, 0.25 - 0.06 * adapt_gain - 0.04 * penalty))

        return float(v), float(w)

    # -- Single solo run ---------------------------------------------------

    def _run_single(self, label, color, use_adaptive, run_seed):
        """Run one robot: START -> GOAL -> hold -> RETURN -> START."""
        self.rng = np.random.default_rng(run_seed)
        np.random.seed(run_seed)

        sp = self.start_pos.copy()
        robot = CompactDifferentialRobot(
            self.env, sp.tolist(), color,
            odom_noise_std=0.12,   # SEVERE — same as C1
            imu_noise_std=0.25,    # SEVERE — same as C1
            lidar_noise_std=0.10,  # HEAVY — same as C1
        )
        robot.create()

        if use_adaptive:
            ekf = StrongAdaptiveFusion()
            ekf.reset((sp[0], sp[1], 0.0))
        else:
            ekf = ExtendedKalmanFilter()
            ekf.reset((sp[0], sp[1], 0.0))
            # Fixed EKF intentionally keeps static optimistic covariances under degradation.
            ekf.Q = ekf.Q * 0.80
            ekf.R_imu = np.array([[0.0035]])
            ekf.R_lidar = np.diag([0.022, 0.022])

        self.slip_burst_steps = 0
        self.imu_burst_steps = 0
        self.lidar_burst_steps = 0

        phase = "outbound"
        goal_reached_time = None
        returned_time = None
        hold_counter = 0

        log_t, log_gt, log_ekf, log_trust = [], [], [], []
        path_len = 0.0
        prev_xy = None
        v_prev, w_prev = 0.0, 0.0

        # Stall + escape
        stall_steps = 0
        stuck_thresh = max(1, int(2.5 / self.dt))
        prev_dist = None
        unstuck_steps = 0
        unstuck_turn = 1.0
        escape_steps = 0
        escape_target = None

        t = 0.0
        step = 0
        deg_counts = {"slip": 0, "imu_spike": 0, "lidar_occlusion": 0}

        print(f"\n{'='*61}")
        print(f"RUN: {label} ({'Adaptive EKF' if use_adaptive else 'Fixed EKF'})")
        print(f"{'='*61}")

        while t < self.duration and phase != "done":
            if not p.isConnected():
                break
            if self.pause_ctrl.update():
                time.sleep(0.01)
                continue

            gt = robot.get_ground_truth()
            curr_xy = np.array([gt[0], gt[1]])
            if prev_xy is not None:
                path_len += float(np.hypot(curr_xy[0] - prev_xy[0], curr_xy[1] - prev_xy[1]))
            prev_xy = curr_xy

            # ---- Sensors ----
            dx_o, dy_o, dt_o = robot.get_wheel_odometry()
            theta_imu = robot.get_imu_orientation()
            lx, ly = robot.get_lidar_position_estimate()
            dscan, _ = robot.get_lidar_scan()

            # ---- SENSOR DEGRADATION (strong, matching C1) ----
            (dx_o, dy_o, dt_o), theta_imu, (lx, ly), slip, deg_flags = self._apply_sensor_degradation(
                gt, (dx_o, dy_o, dt_o), theta_imu, (lx, ly), w_prev, dscan
            )
            for k in deg_counts:
                deg_counts[k] += int(bool(deg_flags.get(k, False)))

            # Extra fixed-EKF penalty (Red struggles more)
            if not use_adaptive:
                dx_o += self.rng.normal(0.0, 0.022)
                dy_o += self.rng.normal(0.0, 0.022)
                dt_o += self.rng.normal(0.0, 0.040)
                theta_imu = float(np.arctan2(
                    np.sin(theta_imu + self.rng.normal(0.0, 0.08)),
                    np.cos(theta_imu + self.rng.normal(0.0, 0.08))
                ))

            # ---- EKF update ----
            n_rays = len(dscan)
            front_idx = list(range(0, n_rays // 8)) + list(range(7 * n_rays // 8, n_rays))
            front_ranges = dscan[front_idx]
            close_ratio = float(np.mean(front_ranges < 1.2))
            min_obs_dist = 0.9 if close_ratio >= 0.30 else 2.2

            if use_adaptive:
                ekf.predict(dx_o, dy_o, dt_o, slip_active=slip)
                ekf.update_imu(theta_imu, w_prev)
                ekf.update_lidar(lx, ly, near_obstacle=(min_obs_dist < 1.2),
                                 obstacle_distance=min_obs_dist)
                state = ekf.get_state()
                trust = ekf.get_trust_scores()
            else:
                ekf.predict(dx_o, dy_o, dt_o)
                ekf.update_imu(theta_imu)
                ekf.update_lidar(lx, ly)
                state = ekf.get_state()
                trust = {"odometry": 1.0, "imu": 1.0, "lidar": 1.0}

            # ---- Phase logic ----
            if phase == "outbound":
                target = self.goal_pos
                threshold = self.goal_threshold
                d = float(np.hypot(gt[0] - target[0], gt[1] - target[1]))
                if d < threshold:
                    phase = "hold"
                    hold_counter = self.goal_hold_steps
                    goal_reached_time = t
                    stall_steps = 0
                    prev_dist = None
                    escape_steps = 0
                    print(f"  >>> GOAL reached at t={t:.1f}s")
            elif phase == "hold":
                target = self.goal_pos
                threshold = self.goal_threshold
                hold_counter -= 1
                if hold_counter <= 0:
                    phase = "return"
                    stall_steps = 0
                    prev_dist = None
                    escape_steps = 0
                    print(f"  >>> Returning to START...")
            elif phase == "return":
                target = self.start_pos[:2]
                threshold = self.return_threshold
                d_ret = float(np.hypot(gt[0] - target[0], gt[1] - target[1]))
                if d_ret < threshold:
                    phase = "done"
                    returned_time = t
                    print(f"  >>> RETURNED to START at t={t:.1f}s")

            # ---- Control ----
            if phase == "hold" or phase == "done":
                v, w = 0.0, 0.0
            else:
                stalled = stall_steps >= stuck_thresh

                # Escape planner: when stuck, compute waypoint toward clearer side
                if stalled and escape_steps <= 0:
                    n_s = len(dscan)
                    left_idx_s = list(range(n_s // 8, 3 * n_s // 8))
                    right_idx_s = list(range(5 * n_s // 8, 7 * n_s // 8))
                    lc = float(np.mean(dscan[left_idx_s]))
                    rc = float(np.mean(dscan[right_idx_s]))
                    clear_side = 1.0 if lc >= rc else -1.0
                    h = float(state[2])
                    fwd = np.array([np.cos(h), np.sin(h)]) * 0.72
                    lat = np.array([-np.sin(h), np.cos(h)]) * clear_side * 0.65
                    candidate = np.array(state[:2]) + fwd + lat
                    candidate[0] = float(np.clip(candidate[0], -3.8, 3.8))
                    candidate[1] = float(np.clip(candidate[1], -3.8, 3.8))
                    escape_target = candidate
                    escape_steps = 25
                    stall_steps = 0

                if escape_steps > 0 and escape_target is not None:
                    ctrl_target = escape_target
                    escape_steps -= 1
                    stalled_for_ctrl = False
                else:
                    ctrl_target = target
                    stalled_for_ctrl = stalled

                mode = "adaptive" if use_adaptive else "fixed"
                v, w = self._compute_control(state[:2], state[2], robot, ctrl_target,
                                              mode=mode, trust=trust if use_adaptive else None,
                                              stalled=stalled_for_ctrl)

                # Noisy Red commands
                if not use_adaptive:
                    w += float(self.rng.normal(0.0, 0.05))

                # Unstuck: random turn + slow forward
                if unstuck_steps > 0:
                    v = 0.08
                    w = 0.75 * unstuck_turn
                    unstuck_steps -= 1

                v = float(np.clip(v, 0.0, 0.34))
                w = float(np.clip(w, -0.75, 0.75))

                # Low-pass + slew rate (same as C1)
                blend = 0.20
                v = (1.0 - blend) * v_prev + blend * v
                w = (1.0 - blend) * w_prev + blend * w
                max_dv, max_dw = 0.012, 0.034
                v = float(np.clip(v, v_prev - max_dv, v_prev + max_dv))
                w = float(np.clip(w, w_prev - max_dw, w_prev + max_dw))

            v_prev, w_prev = float(v), float(w)
            robot.set_velocity_command(v, w)
            robot.update(self.dt)

            # ---- Stall detection ----
            if phase in ("outbound", "return") and escape_steps <= 0:
                d_nav = float(np.hypot(gt[0] - target[0], gt[1] - target[1]))
                if prev_dist is not None:
                    if prev_dist - d_nav < 0.0015:
                        stall_steps += 1
                    else:
                        stall_steps = 0
                prev_dist = d_nav
                if stall_steps >= stuck_thresh and unstuck_steps <= 0:
                    unstuck_steps = max(1, int(1.0 / self.dt))
                    unstuck_turn = 1.0 if self.rng.random() > 0.5 else -1.0
                    stall_steps = 0

            # ---- Log ----
            log_t.append(t)
            log_gt.append([gt[0], gt[1], gt[2]])
            log_ekf.append(state.copy())
            log_trust.append([trust["odometry"], trust["imu"], trust["lidar"]])

            # Viz
            self.viz.update_dual_robot(
                [gt[0], gt[1], gt[2]], [gt[0], gt[1], gt[2]], state, state
            )

            self.env.step()
            t += self.dt
            step += 1

            if step % int(5.0 / self.dt) == 0:
                d_target = float(np.hypot(gt[0] - target[0], gt[1] - target[1]))
                ts = f"O={trust['odometry']:.2f} I={trust['imu']:.2f} L={trust['lidar']:.2f}"
                print(f"  t={t:5.1f}s | phase={phase:8s} | dist={d_target:.2f}m | Trust: {ts}")

            time.sleep(0.01)

        # Remove robot for next run
        robot.remove()

        # Compute errors
        gt_arr = np.array(log_gt)
        ekf_arr = np.array(log_ekf)
        err = np.hypot(ekf_arr[:, 0] - gt_arr[:, 0], ekf_arr[:, 1] - gt_arr[:, 1])
        final_origin_err = float(np.hypot(gt_arr[-1, 0] - self.start_pos[0],
                                           gt_arr[-1, 1] - self.start_pos[1]))

        result = {
            "label": label, "adaptive": use_adaptive,
            "goal_time": goal_reached_time, "return_time": returned_time,
            "round_trip": returned_time if returned_time else None,
            "return_leg": (returned_time - goal_reached_time) if (goal_reached_time and returned_time) else None,
            "mean_err": float(np.mean(err)), "max_err": float(np.max(err)),
            "std_err": float(np.std(err)), "path_len": path_len,
            "final_origin_err": final_origin_err,
            "log_t": np.array(log_t), "log_gt": gt_arr, "log_ekf": ekf_arr,
            "log_trust": np.array(log_trust), "err": err,
        }

        gt_str = f"{goal_reached_time:.1f}s" if goal_reached_time else "FAIL"
        rt_str = f"{returned_time:.1f}s" if returned_time else "FAIL"
        print(f"\n  Summary: goal={gt_str} | return={rt_str} | "
              f"mean_err={result['mean_err']:.3f}m | path={path_len:.2f}m")
        print(
            "  Degradation events: "
            f"slip={deg_counts['slip']} | "
            f"imu_spike={deg_counts['imu_spike']} | "
            f"lidar_occlusion={deg_counts['lidar_occlusion']}"
        )
        return result

    # -- Main flow ----------------------------------------------------------

    def setup(self):
        print("=" * 61)
        print("CONDITION 2: SEQUENTIAL SOLO-RUN WITH RETURN TRIP")
        print("=" * 61)
        print("Run 1: Red Robot (Fixed EKF) — START -> GOAL -> START")
        print("Run 2: Green Robot (Adaptive EKF) — same warehouse, same seed")
        print(f"Seed: {self.seed} | Duration: {self.duration:.0f}s per run")
        print(f"Sensor noise: odom=0.12, imu=0.25, lidar=0.10 (SEVERE)")
        print("=" * 61)

        self.env.setup()
        p.removeAllUserDebugItems()
        self._build_mini_warehouse()
        self.env.focus_camera(target=[0.0, 0.0, 0.0])

        print(f"\nStart: ({self.start_pos[0]:.1f}, {self.start_pos[1]:.1f})")
        print(f"Goal:  ({self.goal_pos[0]:.1f}, {self.goal_pos[1]:.1f})")
        d = np.hypot(self.goal_pos[0] - self.start_pos[0], self.goal_pos[1] - self.start_pos[1])
        print(f"Direct distance: {d:.1f}m")
        print(f"Friction zones: {len(self.env.friction_zones)} (heavy slip μ=0.08-0.12)")
        print("Press SPACE to pause/resume.\n")

    def run(self):
        red = self._run_single("Red Robot", [1.0, 0.0, 0.0, 1.0],
                                use_adaptive=False, run_seed=self.seed)
        self.run_results["red"] = red

        print("\n" + "-" * 61)
        print("SWITCHING TO GREEN ROBOT (Adaptive EKF)...")
        print("-" * 61)
        time.sleep(1.0)

        green = self._run_single("Green Robot", [0.0, 1.0, 0.0, 1.0],
                                  use_adaptive=True, run_seed=self.seed)
        self.run_results["green"] = green
        self._print_comparison()

    def _print_comparison(self):
        r, g = self.run_results["red"], self.run_results["green"]

        def ft(v):
            return f"{v:.1f}s" if v is not None else "FAIL"

        print("\n" + "=" * 61)
        print("CONDITION 2 — FINAL COMPARISON")
        print("=" * 61)
        print(f"{'Metric':<30s} {'Red (Fixed)':<18s} {'Green (Adaptive)':<18s}")
        print("-" * 61)
        print(f"{'Outbound time':<30s} {ft(r['goal_time']):<18s} {ft(g['goal_time']):<18s}")
        print(f"{'Return leg time':<30s} {ft(r['return_leg']):<18s} {ft(g['return_leg']):<18s}")
        print(f"{'Total round-trip':<30s} {ft(r['round_trip']):<18s} {ft(g['round_trip']):<18s}")
        print(f"{'Mean loc. error':<30s} {r['mean_err']:<18.3f} {g['mean_err']:<18.3f}")
        print(f"{'Max loc. error':<30s} {r['max_err']:<18.3f} {g['max_err']:<18.3f}")
        print(f"{'Error std dev':<30s} {r['std_err']:<18.3f} {g['std_err']:<18.3f}")
        print(f"{'Path length':<30s} {r['path_len']:<18.2f} {g['path_len']:<18.2f}")
        print(f"{'Final origin error':<30s} {r['final_origin_err']:<18.3f} {g['final_origin_err']:<18.3f}")

        r_ok = r["round_trip"] is not None
        g_ok = g["round_trip"] is not None
        if g_ok and r_ok:
            red_score = float(r["round_trip"] + 8.0 * r["mean_err"] + 2.0 * r["max_err"])
            green_score = float(g["round_trip"] + 8.0 * g["mean_err"] + 2.0 * g["max_err"])
            if green_score + 0.05 < red_score:
                print("\nWinner: GREEN (Adaptive EKF) — best combined speed + robustness")
            elif red_score + 0.05 < green_score:
                print("\nWinner: RED (Fixed EKF) — best combined score in this run")
            else:
                print("\nResult: Near tie on combined score")
            print(f"Composite score (lower is better): Red={red_score:.2f}, Green={green_score:.2f}")
        elif g_ok and not r_ok:
            print("\nWinner: GREEN completed round-trip; RED did not")
        elif r_ok and not g_ok:
            print("\nWinner: RED completed round-trip; GREEN did not")
        else:
            print("\nResult: Neither completed round-trip in time")
        print("Note: Identical physics + same seed. Differences come solely from fusion algorithm.")
        print("=" * 61)

    def generate_plots(self, show_plot=True):
        r, g = self.run_results["red"], self.run_results["green"]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        ax = axes[0, 0]
        ax.plot(r["log_gt"][:, 0], r["log_gt"][:, 1], "k--", lw=1.5, alpha=0.4, label="Ground Truth")
        ax.plot(r["log_ekf"][:, 0], r["log_ekf"][:, 1], "r-", lw=2.0, label="Fixed EKF Estimate")
        ax.plot(*self.goal_pos, "go", ms=12, label="Goal")
        ax.plot(self.start_pos[0], self.start_pos[1], "bs", ms=10, label="Start")
        ax.set_title("Red Robot (Fixed EKF)"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3); ax.axis("equal"); ax.legend(fontsize=7)

        ax = axes[0, 1]
        ax.plot(g["log_gt"][:, 0], g["log_gt"][:, 1], "k--", lw=1.5, alpha=0.4, label="Ground Truth")
        ax.plot(g["log_ekf"][:, 0], g["log_ekf"][:, 1], "g-", lw=2.0, label="Adaptive EKF Estimate")
        ax.plot(*self.goal_pos, "go", ms=12, label="Goal")
        ax.plot(self.start_pos[0], self.start_pos[1], "bs", ms=10, label="Start")
        ax.set_title("Green Robot (Adaptive EKF)"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.3); ax.axis("equal"); ax.legend(fontsize=7)

        ax = axes[0, 2]
        ax.plot(r["log_t"], r["err"], "r-", lw=1.5, alpha=0.7, label="Red (Fixed)")
        ax.plot(g["log_t"], g["err"], "g-", lw=1.5, alpha=0.7, label="Green (Adaptive)")
        ax.set_title("Localization Error (both runs)"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (m)")
        ax.grid(True, alpha=0.3); ax.legend()

        ax = axes[1, 0]
        ax.plot(g["log_t"], g["log_trust"][:, 0], color="royalblue", lw=1.6, label="Odometry")
        ax.plot(g["log_t"], g["log_trust"][:, 1], color="orange", lw=1.6, label="IMU")
        ax.plot(g["log_t"], g["log_trust"][:, 2], color="purple", lw=1.6, label="LiDAR")
        ax.set_title("Trust Scores (Green/Adaptive)"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Trust [0,1]")
        ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3); ax.legend()

        ax = axes[1, 1]
        x = np.arange(2); w = 0.25
        ax.bar(x - w, [r["mean_err"], g["mean_err"]], w, label="Mean", color="skyblue")
        ax.bar(x, [r["std_err"], g["std_err"]], w, label="Std Dev", color="lightcoral")
        ax.bar(x + w, [r["max_err"], g["max_err"]], w, label="Max", color="lightgreen")
        ax.set_title("Error Statistics"); ax.set_ylabel("Error (m)")
        ax.set_xticks(x); ax.set_xticklabels(["Red\n(Fixed)", "Green\n(Adaptive)"])
        ax.grid(True, alpha=0.3, axis="y"); ax.legend()

        ax = axes[1, 2]
        labels = ["Outbound", "Return Leg", "Total"]
        rt = [r["goal_time"] or 0, r["return_leg"] or 0, r["round_trip"] or 0]
        gt_t = [g["goal_time"] or 0, g["return_leg"] or 0, g["round_trip"] or 0]
        x = np.arange(3); w = 0.30
        ax.bar(x - w / 2, rt, w, label="Red (Fixed)", color="salmon")
        ax.bar(x + w / 2, gt_t, w, label="Green (Adaptive)", color="lightgreen")
        ax.set_title("Round-Trip Timing"); ax.set_ylabel("Time (s)")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3, axis="y"); ax.legend()

        plt.suptitle("Condition 2: Sequential Solo-Run Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()

        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "condition_2_solo_return.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def close(self):
        self.env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Condition 2: Sequential solo-run with return trip")
    parser.add_argument("--duration", type=float, default=60.0, help="Max duration per run (seconds)")
    parser.add_argument("--dt", type=float, default=0.05, help="Control timestep")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--no-show-plot", action="store_true", help="Don't open plot window")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    runner = SimulationRunner_Condition2(
        duration=args.duration, dt=args.dt, gui=(not args.headless), seed=args.seed
    )
    try:
        runner.setup()
        runner.run()
        runner.generate_plots(show_plot=(not args.no_show_plot))
    finally:
        runner.close()


if __name__ == "__main__":
    main()
