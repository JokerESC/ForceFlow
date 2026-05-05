import atexit
import open3d as o3d


class RealsenseEnv:
    def __init__(self, serial: str = None, record: bool = False):
        print(o3d.t.io.RealSenseSensor.list_devices())

        self.rs = o3d.t.io.RealSenseSensor()
        # Configure your RealSense camera model here.
        # serial=None lets open3d auto-select the first available device.
        config = o3d.t.io.RealSenseSensorConfig({
            "serial": serial or "",
            "color_format": "RS2_FORMAT_RGB8",
            "color_resolution": "640,480",
            "depth_format": "RS2_FORMAT_Z16",
            "depth_resolution": "640,480",
            "fps": "30"
        })
        if record:
            self.rs.init_sensor(config, 0, "debug.bag")
            self.rs.start_capture(True)
        else:
            self.rs.init_sensor(config, 0)
            self.rs.start_capture()

        self.intrinsic_matrix = self.rs.get_metadata().intrinsics.intrinsic_matrix
        self.depth_scale = self.rs.get_metadata().depth_scale

        atexit.register(self.cleanup)

    def _get_observation(self) -> dict:
        im_rgbd: o3d.t.geometry.RGBDImage = self.rs.capture_frame(True, True)
        pcd: o3d.t.geometry.PointCloud = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            im_rgbd,
            intrinsics=o3d.core.Tensor(self.intrinsic_matrix, dtype=o3d.core.Dtype.Float32),
            depth_scale=self.depth_scale
        )
        return {
            "im_rgbd": im_rgbd,
            "intrinsic_matrix": self.intrinsic_matrix,
            "depth_scale": self.depth_scale,
            "pcd": pcd
        }

    def reset(self, action=None) -> dict:
        return self._get_observation()

    def step(self, action=None) -> dict:
        return self._get_observation()

    def cleanup(self):
        self.rs.stop_capture()

if __name__ == "__main__":
    import traceback
    import cv2
    import numpy as np

    print(o3d.t.io.RealSenseSensor.list_devices())
    rs_arm = RealsenseEnv(serial=None)  # set your arm-mounted camera serial
    rs_fix = RealsenseEnv(serial=None)  # set your fixed-view camera serial

    try:
        while True:
            rs_arm_obs = rs_arm.step()
            rs_fix_obs = rs_fix.step()
            color_image_arm = np.asarray(rs_arm_obs["im_rgbd"].color)
            color_image_fix = np.asarray(rs_fix_obs["im_rgbd"].color)
            cv2.imshow("rgb_arm", cv2.cvtColor(color_image_arm, cv2.COLOR_RGB2BGR))
            cv2.imshow("rgb_fix", cv2.cvtColor(color_image_fix, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
    except KeyboardInterrupt:
        pass
    except:
        print(traceback.format_exc())
