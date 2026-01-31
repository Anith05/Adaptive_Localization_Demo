"""
Quick verification script to test all imports and basic functionality.
Run this before the main simulation to ensure everything is set up correctly.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("ADAPTIVE LOCALIZATION DEMO - VERIFICATION")
print("=" * 60)
print("\n1. Testing imports...")

try:
    import pybullet
    print("   ✓ PyBullet")
except ImportError as e:
    print(f"   ✗ PyBullet: {e}")
    sys.exit(1)

try:
    import numpy
    print("   ✓ NumPy")
except ImportError as e:
    print(f"   ✗ NumPy: {e}")
    sys.exit(1)

try:
    import matplotlib
    print("   ✓ Matplotlib")
except ImportError as e:
    print(f"   ✗ Matplotlib: {e}")
    sys.exit(1)

print("\n2. Testing project modules...")

try:
    from sim.pybullet_env import PyBulletEnvironment
    print("   ✓ sim.pybullet_env")
except ImportError as e:
    print(f"   ✗ sim.pybullet_env: {e}")
    sys.exit(1)

try:
    from sim.robot import DifferentialDriveRobot
    print("   ✓ sim.robot")
except ImportError as e:
    print(f"   ✗ sim.robot: {e}")
    sys.exit(1)

try:
    from fusion.ekf import ExtendedKalmanFilter
    print("   ✓ fusion.ekf")
except ImportError as e:
    print(f"   ✗ fusion.ekf: {e}")
    sys.exit(1)

try:
    from fusion.ai_trust_model import AITrustModel
    print("   ✓ fusion.ai_trust_model")
except ImportError as e:
    print(f"   ✗ fusion.ai_trust_model: {e}")
    sys.exit(1)

try:
    from fusion.adaptive_fusion import AdaptiveEKF
    print("   ✓ fusion.adaptive_fusion")
except ImportError as e:
    print(f"   ✗ fusion.adaptive_fusion: {e}")
    sys.exit(1)

try:
    from demo.live_visualization import LiveVisualization
    print("   ✓ demo.live_visualization")
except ImportError as e:
    print(f"   ✗ demo.live_visualization: {e}")
    sys.exit(1)

try:
    from demo.pause_controls import PauseControls
    print("   ✓ demo.pause_controls")
except ImportError as e:
    print(f"   ✗ demo.pause_controls: {e}")
    sys.exit(1)

print("\n3. Testing basic instantiation...")

try:
    ekf = ExtendedKalmanFilter()
    print(f"   ✓ EKF created with state: {ekf.get_state()}")
except Exception as e:
    print(f"   ✗ EKF creation failed: {e}")
    sys.exit(1)

try:
    adaptive_ekf = AdaptiveEKF()
    print(f"   ✓ Adaptive EKF created with state: {adaptive_ekf.get_state()}")
except Exception as e:
    print(f"   ✗ Adaptive EKF creation failed: {e}")
    sys.exit(1)

try:
    trust_model = AITrustModel()
    print(f"   ✓ AI Trust Model created")
except Exception as e:
    print(f"   ✗ AI Trust Model creation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL CHECKS PASSED!")
print("=" * 60)
print("\nYour environment is ready to run the simulation.")
print("\nTo start the demo, run:")
print("  python sim\\run_simulation.py")
print("\n" + "=" * 60)
