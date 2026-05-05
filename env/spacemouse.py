from env.spacemouse_expert import SpaceMouseExpert

class SpacemouseAgent():
    def __init__(self):
        self.mouse = SpaceMouseExpert()

    def act(self):
        action, buttons = self.mouse.get_action()
        action[:2] *= 5

        # remap torques to match gripper rotated 90 degrees: swap roll/pitch and negate roll
        original_torque = action[3:] * 0.004
        action[3] = -original_torque[1]  # roll = -pitch
        action[4] = original_torque[0]   # pitch = roll
        action[5] = original_torque[2]

        return action, buttons
