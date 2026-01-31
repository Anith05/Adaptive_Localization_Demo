"""
PyBullet environment with walls, obstacles, and friction zones.
"""
import pybullet as p
import numpy as np


class PyBulletEnvironment:
    """Indoor 2D environment with physics-based failure conditions."""
    
    def __init__(self, gui=True):
        """Initialize PyBullet simulation environment."""
        self.gui = gui
        self.physics_client = None
        self.plane_id = None
        self.wall_ids = []
        self.obstacle_ids = []
        self.friction_zones = []
        
    def setup(self):
        """Set up the simulation environment."""
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)
        
        # Create ground plane
        self.plane_id = p.createCollisionShape(p.GEOM_PLANE)
        p.createMultiBody(0, self.plane_id)
        
        # Create boundary walls (10x10m room)
        self._create_walls()
        
        # Create obstacles
        self._create_obstacles()
        
        # Define low-friction zones (causing wheel slip)
        self._create_friction_zones()
        
    def _create_walls(self):
        """Create boundary walls."""
        wall_thickness = 0.2
        wall_height = 0.5
        room_size = 10.0
        
        # Wall positions: North, South, East, West
        wall_configs = [
            ([0, room_size/2, wall_height/2], [room_size, wall_thickness, wall_height]),  # North
            ([0, -room_size/2, wall_height/2], [room_size, wall_thickness, wall_height]), # South
            ([room_size/2, 0, wall_height/2], [wall_thickness, room_size, wall_height]),  # East
            ([-room_size/2, 0, wall_height/2], [wall_thickness, room_size, wall_height]), # West
        ]
        
        for position, size in wall_configs:
            wall_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[s/2 for s in size]
            )
            wall_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[s/2 for s in size],
                rgbaColor=[0.5, 0.5, 0.5, 1]
            )
            wall_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_shape,
                baseVisualShapeIndex=wall_visual,
                basePosition=position
            )
            self.wall_ids.append(wall_id)
            
    def _create_obstacles(self):
        """Create obstacles in the environment."""
        # Obstacle configurations: (position, size)
        obstacle_configs = [
            ([2, 2, 0.25], [0.8, 0.8, 0.5]),
            ([-2, -2, 0.25], [1.0, 1.0, 0.5]),
            ([3, -1, 0.25], [0.6, 1.2, 0.5]),
            ([-1, 3, 0.25], [1.5, 0.5, 0.5]),
        ]
        
        for position, size in obstacle_configs:
            obs_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[s/2 for s in size]
            )
            obs_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[s/2 for s in size],
                rgbaColor=[0.3, 0.3, 0.3, 1]
            )
            obs_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=obs_shape,
                baseVisualShapeIndex=obs_visual,
                basePosition=position
            )
            self.obstacle_ids.append(obs_id)
            
    def _create_friction_zones(self):
        """Define low-friction zones (simulated as patches on ground)."""
        # Friction zones: (center_x, center_y, radius, friction_coeff)
        # Strategically placed on path from start [-0.3,-4] to goal [3.5,3.5]
        self.friction_zones = [
            (0.5, -2.0, 1.2, 0.08),  # Large slip zone early in path
            (2.0, 0.5, 0.9, 0.12),   # Mid-path slip zone
            (1.0, 1.5, 0.7, 0.15),   # Near goal slip zone
        ]
        
        # Visualize friction zones
        for x, y, radius, friction in self.friction_zones:
            zone_visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=0.01,
                rgbaColor=[0.8, 0.8, 0.3, 0.3]
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=zone_visual,
                basePosition=[x, y, 0.005]
            )
            
    def get_friction_at_position(self, x, y):
        """Get friction coefficient at given position."""
        for fx, fy, radius, friction in self.friction_zones:
            dist = np.sqrt((x - fx)**2 + (y - fy)**2)
            if dist < radius:
                return friction
        return 0.8  # Default friction
    
    def visualize_goal(self, goal_position):
        """Visualize goal position with a marker."""
        goal_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.3,
            length=0.02,
            rgbaColor=[0, 1, 0, 0.7]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=goal_visual,
            basePosition=[goal_position[0], goal_position[1], 0.01]
        )
        
        # Add text label
        p.addUserDebugText(
            "GOAL",
            textPosition=[goal_position[0], goal_position[1], 0.5],
            textColorRGB=[0, 1, 0],
            textSize=2.0,
            lifeTime=0
        )
    
    def check_collision(self, robot_id):
        """Check if robot is in collision."""
        contacts = p.getContactPoints(bodyA=robot_id)
        return len(contacts) > 0
    
    def step(self):
        """Step the simulation."""
        p.stepSimulation()
        
    def close(self):
        """Close the simulation."""
        if self.physics_client is not None:
            p.disconnect()
