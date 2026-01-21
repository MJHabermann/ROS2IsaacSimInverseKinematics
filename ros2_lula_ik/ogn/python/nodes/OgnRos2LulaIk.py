"""
OmniGraph core Python API:
  https://docs.omniverse.nvidia.com/kit/docs/omni.graph/latest/Overview.html

OmniGraph attribute data types:
  https://docs.omniverse.nvidia.com/kit/docs/omni.graph.docs/latest/dev/ogn/attribute_types.html

Collection of OmniGraph code examples in Python:
  https://docs.omniverse.nvidia.com/kit/docs/omni.graph.docs/latest/dev/ogn/ogn_code_samples_python.html

Collection of OmniGraph tutorials:
  https://docs.omniverse.nvidia.com/kit/docs/omni.graph.tutorials/latest/Overview.html
"""
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import geometry_msgs.msg
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
        self._data = None
        self._ros2_node = None
        self._subscription = None
        self._kinematics_solver = None
        self._end_effector_frame = None
        self._joint_names = []
        # call parent class to set up timeline event for custom reset
        super().__init__(initialize=False)

    @property
    def data(self):
        """Get received data, and clean it after reading"""
        tmp = self._data
        self._data = None
        return tmp

    def _callback(self, msg):
        """Function that is called when a message is received by the subscription."""
        # Store the Pose message data
        position = np.array([msg.position.x, msg.position.y, msg.position.z])
        orientation = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        
        # Normalize quaternion to ensure valid rotation (critical for IK!)
        quat_norm = np.linalg.norm(orientation)
        if quat_norm > 1e-6:
            orientation = orientation / quat_norm
        else:
            print(f"[ROS2 IK] WARNING: Invalid quaternion received (near-zero norm)")
            return
        
        print(f"[ROS2 IK] Received message: pos=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}), " +
              f"ori=({orientation[0]:.4f}, {orientation[1]:.4f}, {orientation[2]:.4f}, {orientation[3]:.4f})")
        
        self._data = {
            "position": position,
            "orientation": orientation
        }

    def initialize_ros2(self, node_name, topic_name, queue_size, namespace="", qos_profile="lossy"):
        """Initialize ROS 2 node and subscription.
        
        Args:
            qos_profile: 'lossy' for BEST_EFFORT (drops messages if slow) or 
                         'lossless' for RELIABLE (guarantees delivery)
        """
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
            print(f"[ROS2 IK] Created node: {node_name} with namespace: {namespace}")
        # create ROS 2 subscription for Pose messages
        if not self._subscription:
            # Build full topic name for logging
            full_topic = topic_name
            if namespace:
                full_topic = f"/{namespace}/{topic_name}" if not topic_name.startswith('/') else f"/{namespace}{topic_name}"
            print(f"[ROS2 IK] Subscribing to topic: {full_topic} with QoS: {qos_profile}")
            
            # Create QoS profile based on type
            if qos_profile.lower() == "lossless":
                # Lossless: RELIABLE ensures all messages are delivered
                qos = QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=queue_size
                )
            else:
                # Lossy (default): BEST_EFFORT drops messages if subscriber is slow
                qos = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=min(queue_size, 1)  # Only keep latest to prevent memory buildup
                )
            
            self._subscription = self._ros2_node.create_subscription(
                msg_type=geometry_msgs.msg.Pose,
                topic=topic_name,
                callback=self._callback,
                qos_profile=qos
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
        """Do ROS 2 work to take an incoming message from the topic, if any."""
        if self._ros2_node:
            rclpy.spin_once(self._ros2_node, timeout_sec=timeout_sec)

    def custom_reset(self):
        """On timeline stop, destroy ROS 2 subscription and node."""
        # Use getattr with defaults in case __init__ wasn't fully completed
        ros2_node = getattr(self, '_ros2_node', None)
        subscription = getattr(self, '_subscription', None)
        
        if ros2_node:
            if subscription:
                ros2_node.destroy_subscription(subscription)
            ros2_node.destroy_node()

        self._data = None
        self._ros2_node = None
        self._subscription = None
        self._kinematics_solver = None
        self._end_effector_frame = None
        self._joint_names = []
        self.initialized = False

        try:
            rclpy.try_shutdown()
        except:
            pass


class OgnRos2LulaIk:
    """The OmniGraph node class for Lula Inverse Kinematics with ROS2"""

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

            # Initialize state (ROS 2 node, subscriber, and kinematics solver)
            if not state.initialized:
                # Initialize ROS2 node and subscription
                state.initialize_ros2(
                    node_name="lula_ik_node",
                    topic_name=db.inputs.topicName,
                    queue_size=int(db.inputs.queueSize),
                    namespace=db.inputs.nodeNamespace,
                    qos_profile=db.inputs.qosProfile if db.inputs.qosProfile else "lossy"
                )

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

            # Continuously spin to receive incoming messages (non-blocking)
            try:
                state.spin_once(timeout_sec=0.001)
            except Exception as spin_error:
                db.log_warning(f"Error during ROS2 spin: {spin_error}")

            # Process incoming cartesian command if one was received
            cartesian_data = state.data
            if cartesian_data is not None:
                db.log_info(f"Received target pose - Position: {cartesian_data['position']}, Orientation: {cartesian_data['orientation']}")
                
                # Get initial joint positions for warm start (if provided)
                initial_positions = None
                if len(db.inputs.jointPositions) > 0:
                    initial_positions = np.array(db.inputs.jointPositions)

                # Compute inverse kinematics with retry strategies
                ik_result = state.compute_ik(
                    target_position=cartesian_data["position"],
                    target_orientation=cartesian_data["orientation"],
                    initial_joint_positions=initial_positions
                )

                success = False
                joint_positions = None
                
                if ik_result is not None:
                    joint_positions, success = ik_result
                
                # If full IK failed, try position-only (ignores orientation)
                if not success:
                    db.log_warning("Full pose IK failed, trying position-only...")
                    ik_result = state.compute_ik(
                        target_position=cartesian_data["position"],
                        target_orientation=cartesian_data["orientation"],
                        initial_joint_positions=initial_positions,
                        position_only=True
                    )
                    if ik_result is not None:
                        joint_positions, success = ik_result
                        if success:
                            db.log_info("Position-only IK succeeded (orientation ignored)")

                if success and joint_positions is not None:
                    db.log_info(f"IK solution found: {joint_positions.tolist()}")
                    # Set outputs
                    db.outputs.positionCommand = joint_positions.tolist()
                    db.outputs.jointNames = state._joint_names
                    db.outputs.velocityCommand = [0.0] * len(joint_positions)
                    db.outputs.effortCommand = [0.0] * len(joint_positions)
                    db.outputs.timestamp = 0.0  # TODO: Add proper timestamp

                    # Trigger output execution
                    db.outputs.execOut = omni.graph.core.ExecutionAttributeState.ENABLED
                else:
                    db.log_warning(f"IK solution not found for target pose: pos={cartesian_data['position']}, ori={cartesian_data['orientation']}")

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
