"""
ROS2 Lula Forward Kinematics OmniGraph Node

Computes the end effector pose given joint positions using Lula kinematics solver
and publishes the pose to a ROS2 topic.
"""
import rclpy
import geometry_msgs.msg
import sensor_msgs.msg
import numpy as np
from scipy.spatial.transform import Rotation

import omni.graph.core
from isaacsim.core.nodes import BaseResetNode

from omni.isaac.motion_generation import LulaKinematicsSolver

from ros2_lula_ik.ogn.OgnRos2LulaFkDatabase import OgnRos2LulaFkDatabase


class OgnRos2LulaFkInternalState(BaseResetNode):
    """Convenience class for maintaining per-node state information."""

    def __init__(self):
        """Instantiate the per-node state information"""
        self._ros2_node = None
        self._publisher = None
        self._subscription = None
        self._kinematics_solver = None
        self._end_effector_frame = None
        self._joint_names = []
        self._joint_positions = None
        # call parent class to set up timeline event for custom reset
        super().__init__(initialize=False)

    def _joint_state_callback(self, msg):
        """Callback for joint_states subscription."""
        # Parse joint positions from JointState message
        # The message contains name[], position[], velocity[], effort[]
        if len(msg.position) > 0:
            # Store joint names and positions as a dict for reordering if needed
            self._joint_positions = dict(zip(msg.name, msg.position))

    def initialize_ros2(self, node_name, pub_topic_name, sub_topic_name, queue_size, namespace=""):
        """Initialize ROS 2 node, publisher, and subscriber."""
        try:
            rclpy.init()
        except:
            pass
        # create ROS 2 node with optional namespace
        if not self._ros2_node:
            self._ros2_node = rclpy.create_node(
                node_name=node_name,
                namespace=namespace if namespace else None
            )
        # create ROS 2 publisher for Pose messages
        if not self._publisher:
            self._publisher = self._ros2_node.create_publisher(
                msg_type=geometry_msgs.msg.Pose,
                topic=pub_topic_name,
                qos_profile=queue_size
            )
        # create ROS 2 subscription for JointState messages
        if not self._subscription:
            self._subscription = self._ros2_node.create_subscription(
                msg_type=sensor_msgs.msg.JointState,
                topic=sub_topic_name,
                callback=self._joint_state_callback,
                qos_profile=queue_size
            )

    def initialize_kinematics(self, robot_yaml_path, robot_urdf_path, end_effector_frame=None):
        """Initialize Lula kinematics solver."""
        if not self._kinematics_solver:
            self._kinematics_solver = LulaKinematicsSolver(
                robot_description_path=robot_yaml_path,
                urdf_path=robot_urdf_path
            )
            self._joint_names = self._kinematics_solver.get_joint_names()
            
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

    def publish_pose(self, position, orientation):
        """Publish the pose to the ROS2 topic."""
        if self._publisher:
            pos = np.array(position).flatten()
            
            # Convert rotation matrix to quaternion
            ori = np.array(orientation)
            if ori.shape == (3, 3):
                # Lula returns a 3x3 rotation matrix - convert to quaternion
                rot = Rotation.from_matrix(ori)
                # scipy returns [x, y, z, w], we need [w, x, y, z]
                quat_xyzw = rot.as_quat()
                quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]
            elif ori.size == 4:
                # Already a quaternion
                quat = ori.flatten()
            else:
                print(f"[ROS2 FK] ERROR: Unexpected orientation shape: {ori.shape}")
                return
            
            # Normalize quaternion (should already be normalized, but just in case)
            quat_norm = np.linalg.norm(quat)
            if quat_norm > 1e-6:
                quat = quat / quat_norm
            else:
                print("[ROS2 FK] WARNING: Invalid quaternion (near-zero norm), skipping publish")
                return
            
            msg = geometry_msgs.msg.Pose()
            msg.position.x = float(pos[0])
            msg.position.y = float(pos[1])
            msg.position.z = float(pos[2])
            # Quaternion in [w, x, y, z] format
            msg.orientation.w = float(quat[0])
            msg.orientation.x = float(quat[1])
            msg.orientation.y = float(quat[2])
            msg.orientation.z = float(quat[3])
            self._publisher.publish(msg)

    def spin_once(self, timeout_sec=0.0):
        """Process ROS2 callbacks."""
        if self._ros2_node:
            rclpy.spin_once(self._ros2_node, timeout_sec=timeout_sec)

    def get_ordered_joint_positions(self):
        """Get joint positions ordered according to the kinematics solver's joint names."""
        if self._joint_positions is None or not self._joint_names:
            return None
        
        # Reorder joint positions to match the order expected by the kinematics solver
        ordered_positions = []
        for joint_name in self._joint_names:
            if joint_name in self._joint_positions:
                ordered_positions.append(self._joint_positions[joint_name])
            else:
                # Joint not found in message, cannot compute FK
                return None
        return np.array(ordered_positions)

    def custom_reset(self):
        """On timeline stop, destroy ROS2 publisher, subscriber, and node."""
        if self._ros2_node:
            if self._publisher:
                self._ros2_node.destroy_publisher(self._publisher)
            if self._subscription:
                self._ros2_node.destroy_subscription(self._subscription)
            self._ros2_node.destroy_node()

        self._ros2_node = None
        self._publisher = None
        self._subscription = None
        self._kinematics_solver = None
        self._end_effector_frame = None
        self._joint_names = []
        self._joint_positions = None
        self.initialized = False

        rclpy.try_shutdown()


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

            # Initialize state (ROS2 node, publisher, subscriber, kinematics solver)
            if not state.initialized:
                # Initialize ROS2 node, publisher, and subscriber
                state.initialize_ros2(
                    node_name="lula_fk_node",
                    pub_topic_name=db.inputs.pubTopicName,
                    sub_topic_name=db.inputs.subTopicName,
                    queue_size=int(db.inputs.queueSize),
                    namespace=db.inputs.nodeNamespace
                )

                # Initialize kinematics solver
                state.initialize_kinematics(
                    robot_yaml_path=db.inputs.robotYamlPath,
                    robot_urdf_path=db.inputs.robotUrdfPath,
                    end_effector_frame=db.inputs.endEffectorFrame if db.inputs.endEffectorFrame else None
                )
                db.log_info(f"Initialized FK solver with end effector frame: {state._end_effector_frame}")
                db.log_info(f"Joint names: {state._joint_names}")
                db.log_info(f"Subscribing to: {db.inputs.subTopicName}")
                db.log_info(f"Publishing to: {db.inputs.pubTopicName}")

            # Spin to receive incoming joint state messages
            try:
                state.spin_once(timeout_sec=0.0)
            except Exception as spin_error:
                db.log_warning(f"Error during ROS2 spin: {spin_error}")

            # Get ordered joint positions from the latest joint_states message
            joint_positions = state.get_ordered_joint_positions()
            if joint_positions is None:
                # No joint positions received yet, nothing to publish
                return True
            position, orientation = state.compute_fk(joint_positions)

            if position is not None and orientation is not None:
                # Publish pose to ROS2 topic
                state.publish_pose(position, orientation)
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
