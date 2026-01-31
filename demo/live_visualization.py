"""
Live visualization of robot trajectories in PyBullet.
"""
import pybullet as p
import numpy as np


class LiveVisualization:
    """Real-time visualization of ground truth and estimates for dual robots."""
    
    def __init__(self):
        """Initialize visualization."""
        # Robot A lines (Fixed EKF)
        self.gt_a_line_ids = []
        self.ekf_a_line_ids = []
        
        # Robot B lines (AI-Adaptive EKF)
        self.gt_b_line_ids = []
        self.ekf_b_line_ids = []
        
        # Previous positions
        self.prev_gt_a_pos = None
        self.prev_ekf_a_pos = None
        self.prev_gt_b_pos = None
        self.prev_ekf_b_pos = None
        
        self.line_lifetime = 0  # Permanent lines
        
    def update_dual_robot(self, gt_a_pos, gt_b_pos, ekf_a_pos, ekf_b_pos):
        """
        Update visualization with dual robot positions.
        
        Args:
            gt_a_pos: Robot A ground truth position [x, y, theta]
            gt_b_pos: Robot B ground truth position [x, y, theta]
            ekf_a_pos: Robot A Fixed EKF position [x, y, theta]
            ekf_b_pos: Robot B AI-Adaptive EKF position [x, y, theta]
        """
        # Draw Robot A ground truth path (white dashed - subtle)
        if self.prev_gt_a_pos is not None:
            line_id = p.addUserDebugLine(
                [self.prev_gt_a_pos[0], self.prev_gt_a_pos[1], 0.05],
                [gt_a_pos[0], gt_a_pos[1], 0.05],
                lineColorRGB=[0.8, 0.8, 0.8],
                lineWidth=1,
                lifeTime=self.line_lifetime
            )
            self.gt_a_line_ids.append(line_id)
        
        # Draw Robot B ground truth path (white dashed - subtle)
        if self.prev_gt_b_pos is not None:
            line_id = p.addUserDebugLine(
                [self.prev_gt_b_pos[0], self.prev_gt_b_pos[1], 0.05],
                [gt_b_pos[0], gt_b_pos[1], 0.05],
                lineColorRGB=[0.8, 0.8, 0.8],
                lineWidth=1,
                lifeTime=self.line_lifetime
            )
            self.gt_b_line_ids.append(line_id)
        
        # Draw Robot A Fixed EKF path (RED - bold)
        if self.prev_ekf_a_pos is not None:
            line_id = p.addUserDebugLine(
                [self.prev_ekf_a_pos[0], self.prev_ekf_a_pos[1], 0.12],
                [ekf_a_pos[0], ekf_a_pos[1], 0.12],
                lineColorRGB=[1, 0, 0],
                lineWidth=3,
                lifeTime=self.line_lifetime
            )
            self.ekf_a_line_ids.append(line_id)
        
        # Draw Robot B AI-Adaptive EKF path (GREEN - bold)
        if self.prev_ekf_b_pos is not None:
            line_id = p.addUserDebugLine(
                [self.prev_ekf_b_pos[0], self.prev_ekf_b_pos[1], 0.12],
                [ekf_b_pos[0], ekf_b_pos[1], 0.12],
                lineColorRGB=[0, 1, 0],
                lineWidth=3,
                lifeTime=self.line_lifetime
            )
            self.ekf_b_line_ids.append(line_id)
        
        # Update previous positions
        self.prev_gt_a_pos = gt_a_pos
        self.prev_gt_b_pos = gt_b_pos
        self.prev_ekf_a_pos = ekf_a_pos
        self.prev_ekf_b_pos = ekf_b_pos
    
    def clear(self):
        """Clear all visualization lines."""
        for line_id in (self.gt_a_line_ids + self.ekf_a_line_ids + 
                       self.gt_b_line_ids + self.ekf_b_line_ids):
            try:
                p.removeUserDebugItem(line_id)
            except:
                pass
        
        self.gt_a_line_ids.clear()
        self.ekf_a_line_ids.clear()
        self.gt_b_line_ids.clear()
        self.ekf_b_line_ids.clear()
        
        self.prev_gt_pos = None
        self.prev_ekf_pos = None
        self.prev_ai_pos = None
