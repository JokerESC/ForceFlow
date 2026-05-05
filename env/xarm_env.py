from xarm.wrapper import XArmAPI
import numpy as np
import time
import atexit
import math
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr

class XarmEnv:
    def __init__(self, addr="192.168.1.xxx", action_mode="relative", initial_gripper_position=0):
        print("Connecting")
        self.arm = XArmAPI(
            addr, 
            report_type="real" # for fast force feedback
        )
        print("Connected")
        
        self.action_mode = action_mode
        
        self._clear_error_states()
        
        self._set_gripper()
        # set initial gripper position before enabling force sensor to avoid an unintended open on startup
        self.arm.set_gripper_position(initial_gripper_position, wait=True, speed=5000)
        self._enable_force_sensor()

        atexit.register(self.cleanup)
    
    def _clear_error_states(self, mode=7):
        assert(self.arm)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.arm.set_mode(mode) # https://github.com/xArm-Developer/xArm-Python-SDK/issues/122
        self.arm.set_state(state=0)
        # time.sleep(0.1)
    
    def _enable_force_sensor(self):
        self._clear_error_states()
        time.sleep(0.1)
        code = self.arm.ft_sensor_enable(1)
        if code != 0:
            print(f"Warning: Error code {code} in ft_sensor_enable().")
        time.sleep(0.2)  # wait for sensor to stabilize
        self._clear_error_states()
        time.sleep(0.1)
        code = self.arm.ft_sensor_set_zero()
        if code != 0:
            print(f"Warning: Error code {code} in ft_sensor_set_zero().")
        time.sleep(0.2)  # wait for zero to take effect
        self._clear_error_states()
        
    def _disable_force_sensor(self):
        self.arm.ft_sensor_enable(0)

    def reset_force_sensor_zero(self):
        """Re-zero the force sensor; clears error state afterward."""
        code = self.arm.ft_sensor_set_zero()
        if code != 0:
            print(f"Warning: Error code {code} in ft_sensor_set_zero().")
        time.sleep(0.1)  # wait for zero to take effect
        self._clear_error_states()
        
        return code == 0
    
    def set_action_mode(self, mode):
        """Switch action control mode without reinitializing. mode: 'relative' or 'absolute'."""
        if mode not in ["relative", "absolute"]:
            raise ValueError(f"Invalid action mode: {mode}. Must be 'relative' or 'absolute'.")

        old_mode = self.action_mode
        self.action_mode = mode
        print(f"Action mode switched: {old_mode} -> {mode}")

        # sync goal_pos to current position when switching to absolute mode
        if mode == "absolute" and old_mode == "relative":
            code, cart_pos = self.arm.get_position(is_radian=True)
            if code == 0:
                self.goal_pos = np.array(cart_pos)
                print(f"Updated goal_pos to current position: {self.goal_pos[:3]}")
        
        return self.action_mode
    
    def _set_gripper(self):
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(5000)
        
    def _get_observation(self):
        code, cart_pos = self.arm.get_position(is_radian=True)
        while code != 0:
            print(f"Error code {code} in get_position().")
            self._clear_error_states()
            code, cart_pos = self.arm.get_position(is_radian=True)
        
        code, servo_angle = self.arm.get_servo_angle(is_radian=True)
        while code != 0:
            print(f"Error code {code} in get_servo_angle().")
            self._clear_error_states()
            code, servo_angle = self.arm.get_servo_angle(is_radian=True)
        servo_angle = servo_angle[:6]

        code, gripper_position = self.arm.get_gripper_position()
        while code != 0:
            print(f"Error code {code} in get_gripper_position().")
            self._clear_error_states()
            code, gripper_position = self.arm.get_gripper_position()
        
        return {
            "cart_pos": np.array(cart_pos),
            "servo_angle": np.array(servo_angle),
            "ext_force": np.array(self.arm.ft_ext_force),
            "raw_force": np.array(self.arm.ft_raw_force),
            "goal_pos": np.array(self.goal_pos),
            "gripper_position": np.array(gripper_position),
        }
        
    def reset(self, close_gripper=True):
        print("resetting")

        if close_gripper:
            self.arm.set_gripper_position(0, wait=False, speed=8000)
            time.sleep(1)  # brief delay before moving
        else:
            self.arm.set_gripper_position(840, wait=False, speed=8000)

        self._disable_force_sensor()
        self._clear_error_states(6) # mode 7 is not good, 0/6 is ok
        time.sleep(0.1)
        
        self.goal_pos = np.array([470, 0, 530, 180 / 180 * math.pi, 0, -90 / 180 * math.pi])
        # self.goal_pos = np.array([470, 0, 530, 180 / 180 * math.pi, 0, 0])
        
        while True:
            # use position control to fast reset
            code = self.arm.set_servo_angle(angle=[0, 0, -math.pi/2, 0, math.pi/2, math.pi/2, 0], speed=0.5, is_radian=True)
            # code = self.arm.set_servo_angle(angle=[0, 0, -math.pi/2, 0, math.pi/2, 0, 0], speed=0.5, is_radian=True)
        
            code, curr_pos = self.arm.get_position(is_radian=True)
            time.sleep(0.02)
            
            pos_dist = math.dist(self.goal_pos[:3], curr_pos[:3])
            rot_dist = pr.quaternion_dist(
                pr.quaternion_from_extrinsic_euler_xyz(self.goal_pos[3:6]),
                pr.quaternion_from_extrinsic_euler_xyz(curr_pos[3:6])
            )
            
            print(f"resetting, pos_dist {pos_dist} rot_dist {rot_dist}, curr_pos {curr_pos}")
            
            if (pos_dist < 20 and rot_dist < 0.02):
                break

        self._clear_error_states()
        time.sleep(0.2)
        self._enable_force_sensor()

        print("resetting done")

        return self._get_observation()

    
    def step(self, action, gripper_action=None, speed=1000) -> dict[str, np.ndarray]:
        if self.action_mode == "relative":
            self.goal_pos += action
        else:
            self.goal_pos = action
        
        code = self.arm.set_position(
            x=self.goal_pos[0],
            y=self.goal_pos[1], 
            z=self.goal_pos[2], 
            roll=self.goal_pos[3], 
            pitch=self.goal_pos[4], 
            yaw=self.goal_pos[5], 
            speed=speed, wait=False, is_radian=True)
        while code != 0:
            print(f"Error code {code} in set_position().")
            self._clear_error_states()
            code = self.arm.set_position(
                x=self.goal_pos[0],
                y=self.goal_pos[1], 
                z=self.goal_pos[2], 
                roll=self.goal_pos[3], 
                pitch=self.goal_pos[4], 
                yaw=self.goal_pos[5], 
                speed=speed, wait=False, is_radian=True)
        
        if gripper_action is not None:            
            code = self.arm.set_gripper_position(gripper_action, wait=False)

            while code != 0:
                print(f"Error code {code} in set_gripper_position().")
                self._clear_error_states()
                code = self.arm.set_gripper_position(gripper_action, wait=False)

        return self._get_observation()

    def cleanup(self):
        self._disable_force_sensor()
        self.arm.disconnect()
        
if __name__ == "__main__":
    env = XarmEnv()
    env.reset()
    
    while True:
        env.step([0, 0, 0, 0, 0, 0])
        time.sleep(0.1)
