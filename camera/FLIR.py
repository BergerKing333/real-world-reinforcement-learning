

#!/home/adam/miniforge3/envs/camera/bin/python3
# FLIR ROS2 Node — Jazzy
# GigE Vision camera at 169.255.0.6
#
# Modes
# -----
#   default    Grabs frames continuously; holds latest in memory.
#              Send a folder path to /flir/save_request to save the current
#              frame as a raw BGR8 .bin file.  The full file path is published
#              immediately to /flir/save_response (and /flir/image_path) before
#              the disk write completes so callers aren't blocked.
#
#   --save-all Saves every frame to /tmp/flir/<run_timestamp>/ as raw BGR8 .bin.
#              Each path is published to /flir/image_path immediately after the
#              file name is chosen (write happens in a background thread).

import os
import socket
import struct
import time
import concurrent.futures
import argparse

import PySpin
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

CAMERA_IP   = '169.255.0.6'  # GigE camera address

EXPOSURE_US = 200000.0   # microseconds
GAIN_DB     = 0.0       # dB
GAMMA       = 1.0       # 1.0 = off
PUBLISH_HZ  = 60        # camera hardware frame rate AND ROS publish rate
WRITE_THREADS = 4       # parallel disk-write workers


class FLIRCameraNode(Node):

    def __init__(self, save_all: bool):
        super().__init__('flir_camera')

        self._save_all     = save_all
        self._latest_frame = None   # (ndarray copy, h, w) — most recent frame
        self._frame_idx    = 0
        self._executor     = concurrent.futures.ThreadPoolExecutor(max_workers=WRITE_THREADS)

        # Always publish saved-file paths here (for the viewer)
        self._pub = self.create_publisher(String, 'flir/image_path', 10)

        if save_all:
            run_ts = time.strftime('%Y%m%d_%H%M%S')
            self._run_dir = f'/tmp/flir/{run_ts}'
            os.makedirs(self._run_dir, exist_ok=True)
            self.get_logger().info(f'--save-all: saving every frame to {self._run_dir}')
        else:
            self._run_dir = None
            # On-demand: subscribe to save requests, respond with the file path
            self._save_resp_pub = self.create_publisher(String, 'flir/save_response', 10)
            self._save_sub = self.create_subscription(
                String, 'flir/save_request', self._on_save_request, 10
            )
            self.get_logger().info(
                'On-demand mode: publish a folder path to /flir/save_request '
                'to capture a frame. Path is returned on /flir/save_response.'
            )

        # --- Spinnaker init ---
        self._system   = PySpin.System.GetInstance()
        self._cam_list = self._system.GetCameras()

        if self._cam_list.GetSize() == 0:
            self.get_logger().fatal('No FLIR cameras detected.')
            raise RuntimeError('No FLIR cameras found')

        self._cam = self._cam_list[0]
        self._processor = PySpin.ImageProcessor()
        self._processor.SetColorProcessing(
            PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR
        )

        self._cam_init_with_force_ip()
        self._cam.GevSCPSPacketSize.SetValue(1400)

        # Disable any external / software trigger left by SpinView
        self._cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)

        # Internal frame-rate generator
        self._cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self._cam.ExposureTime.SetValue(EXPOSURE_US)

        for node_name in ('AcquisitionFrameRateEnable', 'AcquisitionFrameRateEnabled'):
            node = self._cam.GetNodeMap().GetNode(node_name)
            if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
                PySpin.CBooleanPtr(node).SetValue(True)
                break

        hw_fps = min(PUBLISH_HZ, self._cam.AcquisitionFrameRate.GetMax())
        if hw_fps < PUBLISH_HZ:
            self.get_logger().warn(
                f'Requested {PUBLISH_HZ} fps exceeds camera maximum '
                f'({self._cam.AcquisitionFrameRate.GetMax():.2f} fps) '
                f'at {EXPOSURE_US} µs exposure — clamping to {hw_fps:.2f} fps.'
            )
        self._cam.AcquisitionFrameRate.SetValue(hw_fps)

        self._cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        self._cam.Gain.SetValue(GAIN_DB)
        self._cam.GammaEnable.SetValue(GAMMA != 1.0)
        if GAMMA != 1.0:
            self._cam.Gamma.SetValue(GAMMA)

        self._cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self._cam.BeginAcquisition()

        self.get_logger().info(f'FLIR camera ready @ {hw_fps:.2f} Hz')
        self._timer = self.create_timer(1.0 / hw_fps, self._grab_frame)

    # ------------------------------------------------------------------
    def _cam_init_with_force_ip(self):
        """Call cam.Init(), recovering automatically if SpinView left a bad subnet IP."""
        try:
            self._cam.Init()
            return
        except PySpin.SpinnakerException as ex:
            if '-1015' not in str(ex) and 'subnet' not in str(ex).lower():
                raise
            self.get_logger().warn(
                f'Subnet mismatch detected ({ex}). '
                'Issuing GevCP ForceIP to hijack camera...'
            )

        nodemap_tl = self._cam.GetTLDeviceNodeMap()
        ip_int = struct.unpack('!I', socket.inet_aton(CAMERA_IP))[0]

        addr_node = PySpin.CIntegerPtr(nodemap_tl.GetNode('GevDeviceForceIPAddress'))
        mask_node = PySpin.CIntegerPtr(nodemap_tl.GetNode('GevDeviceForceSubnetMask'))
        gw_node   = PySpin.CIntegerPtr(nodemap_tl.GetNode('GevDeviceForceGateway'))
        cmd_node  = PySpin.CCommandPtr(nodemap_tl.GetNode('GevDeviceForceIP'))

        if not all(
            PySpin.IsAvailable(n) and PySpin.IsWritable(n)
            for n in (addr_node, mask_node, gw_node, cmd_node)
        ):
            raise RuntimeError(
                'GevDeviceForceIP nodes not available — '
                'manually fix the camera IP or adapter subnet.'
            )

        addr_node.SetValue(ip_int)
        mask_node.SetValue(0xFFFF0000)
        gw_node.SetValue(0)
        cmd_node.Execute()
        self.get_logger().info(f'ForceIP sent to {CAMERA_IP} — waiting for re-enumeration...')
        time.sleep(2.0)

        self._cam_list.Clear()
        self._cam_list = self._system.GetCameras()
        if self._cam_list.GetSize() == 0:
            raise RuntimeError('No cameras found after ForceIP')
        self._cam = self._cam_list[0]
        self._cam.Init()

    # ------------------------------------------------------------------
    def _grab_frame(self):
        """Grab one frame from the camera, update latest_frame, and save if --save-all."""
        try:
            image_result = self._cam.GetNextImage(1000)
            if image_result.IsIncomplete():
                image_result.Release()
                return

            frame = self._processor.Convert(
                image_result, PySpin.PixelFormat_BGR8
            ).GetNDArray()
            image_result.Release()

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # Always keep a copy of the latest frame for on-demand saves
            self._latest_frame = (frame.copy(), h, w)

            if self._save_all:
                path = f'{self._run_dir}/{self._frame_idx:08d}.bin'
                self._frame_idx += 1
                self._write_raw_async(frame.copy(), path, w, h,
                                      publish_to=[self._pub])

        except PySpin.SpinnakerException as ex:
            self.get_logger().error(f'Spinnaker error: {ex}')

    # ------------------------------------------------------------------
    def _on_save_request(self, msg: String):
        """Save the latest frame to the requested folder; respond with the path immediately."""
        if self._latest_frame is None:
            self.get_logger().warn('Save requested but no frame available yet.')
            return

        frame, h, w = self._latest_frame
        folder = msg.data.strip()

        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as ex:
            self.get_logger().error(f'Cannot create save directory {folder}: {ex}')
            return

        # Use nanosecond timestamp for a unique, sortable filename
        path = f'{folder}/{time.monotonic_ns()}.bin'

        # Publish path immediately — write happens in the background
        self._write_raw_async(frame.copy(), path, w, h, publish_to=[self._save_resp_pub])

        # self._publish_path(path, w, h, extra_pubs=[self._save_resp_pub])


    # ------------------------------------------------------------------
    def _write_raw_async(self, frame, path: str, w: int, h: int, publish_to: list):
        """Write raw BGR8 bytes to disk in a background thread.
        publish_to: list of publishers to notify after the write completes."""
        pub_list = list(publish_to)

        def _do_write(f, p, publishers, width, height):
            f.tofile(p)
            for pub in publishers:
                self._publish_path(p, width, height, extra_pubs=[], publisher=pub)

        self._executor.submit(_do_write, frame, path, pub_list, w, h)

    # ------------------------------------------------------------------
    def _publish_path(self, path: str, w: int, h: int,
                      extra_pubs: list, publisher=None):
        """Publish  path|W|H  to the given publisher(s)."""
        m = String()
        m.data = f'{path}|{w}|{h}'
        if publisher:
            publisher.publish(m)
        for pub in extra_pubs:
            pub.publish(m)

    # ------------------------------------------------------------------
    def destroy_node(self):
        self.get_logger().info('Shutting down FLIR camera node.')
        self._executor.shutdown(wait=True)
        try:
            self._cam.EndAcquisition()
        except Exception:
            pass
        self._cam.DeInit()
        del self._cam
        self._cam_list.Clear()
        self._system.ReleaseInstance()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description='FLIR GigE ROS2 camera node')
    parser.add_argument('--save-all', action='store_true',
                        help='Save every frame as raw BGR8 .bin to /tmp/flir/<timestamp>/')
    known, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = FLIRCameraNode(save_all=known.save_all)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

