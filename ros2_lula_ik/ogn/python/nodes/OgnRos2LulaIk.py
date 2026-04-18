"""
OmniGraph Lula Inverse Kinematics Node

Computes inverse kinematics for target poses using Lula kinematics solver.
Inputs and outputs are delivered via OmniGraph attributes.
"""
import numpy as np

import omni.graph.core
from isaacsim.core.nodes import BaseResetNode

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.motion_generation import LulaKinematicsSolver

from ros2_lula_ik.ogn.OgnRos2LulaIkDatabase import OgnRos2LulaIkDatabase


class OgnRos2LulaIkInternalState(BaseResetNode):
    """Convenience class for maintaining per-node state information.

    It inherits from ``BaseResetNode`` to do custom reset operation when the timeline is stopped."""

    def __init__(self):
        """Instantiate the per-node state information"""
        self._kinematics_solver = None
        self._end_effector_frame = None
        self._joint_names = []
        # call parent class to set up timeline event for custom reset
        super().__init__(initialize=False)

    def initialize_kinematics(self, robot_yaml_path, robot_urdf_path, end_effector_frame=None):
        """Initialize Lula kinematics solver."""
        if not self._kinematics_solver:
            self._kinematics_solver = LulaKinematicsSolver(
                robot_description_path=robot_yaml_path,
                urdf_path=robot_urdf_path
            )
            all_joint_names = self._kinematics_solver.get_joint_names()
            # Only use the first 6 joints
            self._joint_names = all_joint_names[:6]
            # Get end effector frame from solver if not explicitly provided
            if end_effector_frame:
                self._end_effector_frame = end_effector_frame
            else:
                # Try to get available frames from the robot description YAML
                available_frames = None
                # Try different method names depending on API version
                for method_name in ['get_valid_frames', 'get_all_frame_names', 'get_frame_names']:
                    if hasattr(self._kinematics_solver, method_name):
                        available_frames = getattr(self._kinematics_solver, method_name)()
                        break
                
                if available_frames:
                    available_frames = list(available_frames)
                    available_frames.sort()
                    self._end_effector_frame = available_frames[-1]
                else:
                    raise RuntimeError("No valid frames found in robot description YAML. Please specify endEffectorFrame manually.")
        self.initialized = True

    def compute_ik(self, target_position, target_orientation, initial_joint_positions=None, position_only=False):
        """Compute inverse kinematics for the target pose.
        
        Args:
            target_position: [x, y, z] target position
            target_orientation: [w, x, y, z] target quaternion
            initial_joint_positions: Optional warm start joint positions
            position_only: If True, ignore orientation constraint (easier to solve)
        """
        if not self._kinematics_solver:
            return None

        # Try with full pose constraint first
        orientation_to_use = None if position_only else target_orientation
        
        ik_result = self._kinematics_solver.compute_inverse_kinematics(
            frame_name=self._end_effector_frame,
            target_position=target_position,
            target_orientation=orientation_to_use,
            warm_start=initial_joint_positions
        )
        
        return ik_result

    def spin_once(self, timeout_sec=0.01):
        """Placeholder for compatibility - not needed with OmniGraph inputs."""
        pass

    def custom_reset(self):
        """On timeline stop, cleanup."""
        self._kinematics_solver = None
        self._end_effector_frame = None
        self._joint_names = []
        self.initialized = False


class OgnRos2LulaIk:
    """The OmniGraph node class for Lula Inverse Kinematics"""

    @staticmethod
    def internal_state():
        """Get per-node state information."""
        return OgnRos2LulaIkInternalState()

    @staticmethod
    def compute(db) -> bool:
        """Compute the output based on inputs and internal state."""
        state = db.per_instance_state

        try:
            # Validate required inputs
            if not db.inputs.robotYamlPath or not db.inputs.robotUrdfPath:
                db.log_warning("Robot YAML and URDF paths are required for kinematics solver")
                return False

            # Initialize state (kinematics solver)
            if not state.initialized:
                # Optionally add robot to stage
                if db.inputs.robotUsdPath and db.inputs.robotPrimPath:
                    try:
                        # Ensure prim path is absolute
                        prim_path = db.inputs.robotPrimPath
                        if not prim_path.startswith('/'):
                            prim_path = '/World/' + prim_path
                        add_reference_to_stage(db.inputs.robotUsdPath, prim_path)
                    except Exception as e:
                        db.log_warning(f"Could not add robot to stage: {e}")

                # Initialize kinematics solver (end effector frame is optional, will be read from YAML if not provided)
                state.initialize_kinematics(
                    robot_yaml_path=db.inputs.robotYamlPath,
                    robot_urdf_path=db.inputs.robotUrdfPath,
                    end_effector_frame=db.inputs.endEffectorFrame if db.inputs.endEffectorFrame else None
                )
                db.log_info(f"Initialized kinematics solver with end effector frame: {state._end_effector_frame}")
                db.log_info(f"Joint names: {state._joint_names}")

            # Get target pose from OmniGraph inputs (single array: [x, y, z, qw, qx, qy, qz])
            target_pose = None
            
            try:
                if hasattr(db.inputs, 'targetPose') and db.inputs.targetPose is not None:
                    input_data = db.inputs.targetPose
                    
                    # Handle different input formats
                    # Could be: list, numpy array, or ROS2 Float64MultiArray message
                    if isinstance(input_data, dict) and 'data' in input_data:
                        # ROS2 Float64MultiArray format: {"data": [...]}
                        input_data = input_data['data']
                    elif hasattr(input_data, 'data'):
                        # ROS2 message object with .data attribute
                        input_data = input_data.data
                    
                    target_pose = np.array(input_data, dtype=np.float64)
                    db.log_info(f"Parsed target pose: {target_pose}")
            except (TypeError, ValueError, AttributeError) as e:
                db.log_warning(f"Invalid target pose format: {type(db.inputs.targetPose)} - {str(e)}")
                db.log_warning(f"Raw input: {db.inputs.targetPose}")
                return True
            
            if target_pose is None or target_pose.size < 7:
                db.log_warning(f"Target pose must have 7 values [x, y, z, qw, qx, qy, qz], got {target_pose.size if target_pose is not None else 0}")
                return True
            
            # Split into position and orientation
            target_position = target_pose[0:3]
            target_orientation = target_pose[3:7]

            # Normalize quaternion to ensure valid rotation (critical for IK!)
            quat_norm = np.linalg.norm(target_orientation)
            if quat_norm > 1e-6:
                target_orientation = target_orientation / quat_norm
            else:
                db.log_warning("Invalid quaternion received (near-zero norm)")
                return True
            
            db.log_info(f"Received target pose - Position: {target_position}, Orientation: {target_orientation}")
            
            # Get initial joint positions for warm start (if provided)
            initial_positions = None
            if len(db.inputs.jointPositions) > 0:
                # Only use the first 6 positions
                initial_positions = np.array(db.inputs.jointPositions[:6])

            # Compute inverse kinematics with retry strategies
            db.log_info(f"Attempting IK with target position: {target_position}")
            db.log_info(f"Target orientation (quaternion): {target_orientation}")
            
            ik_result = state.compute_ik(
                target_position=target_position,
                target_orientation=target_orientation,
                initial_joint_positions=initial_positions
            )

            success = False
            joint_positions = None
            
            if ik_result is not None:
                joint_positions, success = ik_result
                db.log_info(f"Full pose IK result - Success: {success}")
            else:
                db.log_error("IK solver returned None")
            
            # If full IK failed, try position-only (ignores orientation)
            if not success:
                db.log_warning("Full pose IK failed, trying position-only approach...")
                ik_result = state.compute_ik(
                    target_position=target_position,
                    target_orientation=target_orientation,
                    initial_joint_positions=initial_positions,
                    position_only=True
                )
                if ik_result is not None:
                    joint_positions, success = ik_result
                    db.log_info(f"Position-only IK result - Success: {success}")
                    if success:
                        db.log_info("Position-only IK succeeded (orientation ignored)")
                else:
                    db.log_error("Position-only IK solver also returned None")

            if success and joint_positions is not None:
                db.log_info(f"IK solution found: {joint_positions.tolist()}")
                # Set outputs
                db.outputs.positionCommand = joint_positions.tolist()
                db.outputs.jointNames = state._joint_names
                db.outputs.velocityCommand = [0.0] * len(joint_positions)
                db.outputs.effortCommand = [0.0] * len(joint_positions)
                db.outputs.timestamp = 0.0

                # Trigger output execution
                db.outputs.execOut = omni.graph.core.ExecutionAttributeState.ENABLED
            else:
                db.log_warning(f"IK solution not found for target pose: pos={target_position}, ori={target_orientation}")

        except Exception as e:
            db.log_error(f"Computation error: {e}")
            return False
        return True

    @staticmethod
    def release(node):
        """Release per-node state information."""
        try:
            state = OgnRos2LulaIkDatabase.per_instance_internal_state(node)
        except Exception:
            return
        # reset state
        state.reset()
        state.initialized = False
