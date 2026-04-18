#!/usr/bin/env python3
"""
Direct IK solver test - bypasses ROS2 for debugging
Usage: python3 test_ik_direct.py <yaml_path> <urdf_path>
"""
import sys
import numpy as np

# You may need to set up Isaac Sim Python path
try:
    from omni.isaac.motion_generation import LulaKinematicsSolver
except ImportError:
    print("ERROR: Cannot import LulaKinematicsSolver. Make sure you're running from Isaac Sim Python environment.")
    print("Try: cd /isaacsim && source setup_python_env.sh && python3 test_ik_direct.py <yaml> <urdf>")
    sys.exit(1)

if len(sys.argv) < 3:
    print("Usage: python3 test_ik_direct.py <yaml_path> <urdf_path> [end_effector_frame]")
    print("\nExample:")
    print("  python3 test_ik_direct.py ./robot.yaml ./robot.urdf tool0")
    sys.exit(1)

yaml_path = sys.argv[1]
urdf_path = sys.argv[2]
ee_frame = sys.argv[3] if len(sys.argv) > 3 else None

print(f"Loading kinematics solver...")
print(f"  YAML: {yaml_path}")
print(f"  URDF: {urdf_path}")
print(f"  End-effector frame: {ee_frame or '(auto-detect)'}")

try:
    solver = LulaKinematicsSolver(
        robot_description_path=yaml_path,
        urdf_path=urdf_path
    )
    print("✓ Solver initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize solver: {e}")
    sys.exit(1)

# Get joint info
joint_names = solver.get_joint_names()
print(f"\nRobot DOF: {len(joint_names)} joints")
print(f"Joints: {joint_names}")

# Get end effector frame
if ee_frame:
    ee_frame_to_use = ee_frame
else:
    try:
        for method_name in ['get_valid_frames', 'get_all_frame_names', 'get_frame_names']:
            if hasattr(solver, method_name):
                frames = list(getattr(solver, method_name)())
                frames.sort()
                ee_frame_to_use = frames[-1]
                print(f"Available frames: {frames}")
                break
    except:
        ee_frame_to_use = None
        
if ee_frame_to_use:
    print(f"End-effector frame: {ee_frame_to_use}")
else:
    print("✗ Could not determine end-effector frame")
    sys.exit(1)

# Test simple target (home-like position)
print("\n" + "="*60)
print("TEST 1: Simple home-like position")
print("="*60)

target_pos = np.array([0.5, 0.0, 0.5])
target_ori = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion [w, x, y, z]

print(f"Target: position={target_pos}, orientation={target_ori}")

try:
    ik_result = solver.compute_inverse_kinematics(
        frame_name=ee_frame_to_use,
        target_position=target_pos,
        target_orientation=target_ori,
        warm_start=None
    )
    if ik_result:
        sol, success = ik_result
        if success:
            print(f"✓ IK SUCCESS: {sol}")
        else:
            print(f"✗ IK FAILED: No solution found")
    else:
        print(f"✗ Solver returned None")
except Exception as e:
    print(f"✗ IK computation error: {e}")

# Test position-only
print("\n" + "="*60)
print("TEST 2: Position-only (no orientation constraint)")
print("="*60)

try:
    ik_result = solver.compute_inverse_kinematics(
        frame_name=ee_frame_to_use,
        target_position=target_pos,
        target_orientation=None,  # No orientation constraint
        warm_start=None
    )
    if ik_result:
        sol, success = ik_result
        if success:
            print(f"✓ Position-only IK SUCCESS: {sol}")
        else:
            print(f"✗ Position-only IK FAILED")
    else:
        print(f"✗ Solver returned None")
except Exception as e:
    print(f"✗ Position-only IK error: {e}")

# Test FK at home (all zeros) to verify robot is loadable
print("\n" + "="*60)
print("TEST 3: Forward Kinematics at home (all zeros)")
print("="*60)

home_joints = np.zeros(len(joint_names))
print(f"Home joint configuration: {home_joints}")

try:
    pos, ori = solver.compute_forward_kinematics(
        frame_name=ee_frame_to_use,
        joint_positions=home_joints
    )
    print(f"✓ FK SUCCESS")
    print(f"  Position: {pos}")
    print(f"  Orientation (matrix):\n{ori}")
except Exception as e:
    print(f"✗ FK error: {e}")

print("\n" + "="*60)
print("Done! Check results above to debug your configuration.")
print("="*60)
