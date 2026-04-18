"""
OmniGraph Lula Forward Kinematics Node

Computes the end effector pose given joint positions using Lula kinematics solver.
Inputs and outputs are delivered via OmniGraph attributes.
"""
import numpy as np

import omni.graph.core
from isaacsim.core.nodes import BaseResetNode

from omni.isaac.motion_generation import LulaKinematicsSolver

from ros2_lula_ik.ogn.OgnRos2LulaFkDatabase import OgnRos2LulaFkDatabase


class OgnRos2LulaFkInternalState(BaseResetNode):
    """Convenience class for maintaining per-node state information."""

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

    def compute_fk(self, joint_positions):
        """Compute forward kinematics for the given joint positions."""
        if not self._kinematics_solver:
            return None, None

        # Compute FK solution
        position, orientation = self._kinematics_solver.compute_forward_kinematics(
            frame_name=self._end_effector_frame,
            joint_positions=joint_positions
        )
        
        return position, orientation

    def custom_reset(self):
        """On timeline stop, cleanup."""
        self._kinematics_solver = None
        self._end_effector_frame = None
        self._joint_names = []
        self.initialized = False


class OgnRos2LulaFk:
    """The OmniGraph node class for ROS2 Lula Forward Kinematics"""

    @staticmethod
    def internal_state():
        """Get per-node state information."""
        return OgnRos2LulaFkInternalState()

    @staticmethod
    def compute(db) -> bool:
        """Compute the output based on inputs and internal state."""
        state = db.per_instance_state

        try:
            # Validate required inputs
            if not db.inputs.robotYamlPath or not db.inputs.robotUrdfPath:
                db.log_warning("Robot YAML and URDF paths are required for kinematics solver")
                return False

            # Initialize state (kinematics solver only)
            if not state.initialized:
                # Initialize kinematics solver
                state.initialize_kinematics(
                    robot_yaml_path=db.inputs.robotYamlPath,
                    robot_urdf_path=db.inputs.robotUrdfPath,
                    end_effector_frame=db.inputs.endEffectorFrame if db.inputs.endEffectorFrame else None
                )
                db.log_info(f"Initialized FK solver with end effector frame: {state._end_effector_frame}")
                db.log_info(f"Joint names: {state._joint_names}")

            # Get joint positions from OmniGraph inputs
            if len(db.inputs.jointPositions) == 0:
                db.log_warning("Joint positions input is required")
                return True
            
            # Only use the first 6 joint positions
            joint_positions = np.array(db.inputs.jointPositions[:6])
            
            # Validate joint positions count matches robot joints
            if len(state._joint_names) > 0 and len(joint_positions) != len(state._joint_names):
                db.log_error(
                    f"Joint positions mismatch: expected {len(state._joint_names)} joints "
                    f"({state._joint_names}), but got {len(joint_positions)} values"
                )
                return True
            
            # Compute FK
            position, orientation = state.compute_fk(joint_positions)

            if position is not None and orientation is not None:
                # Convert orientation to quaternion if needed
                ori = np.array(orientation)
                if ori.shape == (3, 3):
                    # Lula returns a 3x3 rotation matrix - convert to quaternion
                    from scipy.spatial.transform import Rotation
                    rot = Rotation.from_matrix(ori)
                    # scipy returns [x, y, z, w], we need [w, x, y, z]
                    quat_xyzw = rot.as_quat()
                    quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]
                elif ori.size == 4:
                    # Already a quaternion
                    quat = ori.flatten()
                else:
                    db.log_error(f"Unexpected orientation shape: {ori.shape}")
                    return False
                
                # Normalize quaternion
                quat_norm = np.linalg.norm(quat)
                if quat_norm > 1e-6:
                    quat = quat / quat_norm
                else:
                    db.log_warning("Invalid quaternion (near-zero norm), setting to identity")
                    quat = np.array([1.0, 0.0, 0.0, 0.0])
                
                # Set OmniGraph outputs - combine position and orientation into single array
                pos = np.array(position).flatten()
                end_effector_pose = np.concatenate([pos, quat])
                db.outputs.endEffectorPose = end_effector_pose.tolist()
                db.log_info(f"FK computed: endEffectorPose={end_effector_pose.tolist()}")
            else:
                db.log_warning("Failed to compute FK - solver returned None")

        except Exception as e:
            db.log_error(f"Computation error: {e}")
            return False
        return True

    @staticmethod
    def release(node):
        """Release per-node state information."""
        try:
            state = OgnRos2LulaFkDatabase.per_instance_internal_state(node)
        except Exception:
            return
        # reset state
        state.reset()
        state.initialized = False
