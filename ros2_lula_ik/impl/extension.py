"""
Extension entry point for Lula Kinematics OmniGraph nodes extension.
"""
import omni.ext


class ROS2LulaIkExtension(omni.ext.IExt):
    """Extension class for Lula Kinematics OmniGraph nodes."""

    def on_startup(self, ext_id):
        """Called when the extension is being loaded."""
        pass

    def on_shutdown(self):
        """Called when the extension is being unloaded."""
        pass
