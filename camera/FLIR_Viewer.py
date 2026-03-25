#!/home/adam/miniforge3/envs/camera/bin/python3
# FLIR Viewer — ROS2 Jazzy
# Subscribes to /flir/image_path, loads saved PNGs from disk.
# Image shown in an OpenCV window; stats printed live in the terminal.

import os
import time
import curses
import collections

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

TOPIC       = '/flir/image_path'
WINDOW      = 'FLIR Viewer  [q / Esc = quit]'
STATS_ALPHA = 0.05   # EMA smoothing (lower = smoother)

# ANSI colour helpers (curses colour-pair indices)
_GREEN  = 1
_YELLOW = 2
_RED    = 3
_CYAN   = 4
_WHITE  = 5
_GREY   = 6


class FLIRViewer(Node):

    def __init__(self, stdscr):
        super().__init__('flir_viewer')

        # --- curses setup ---
        self._scr = stdscr
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(_GREEN,  curses.COLOR_GREEN,  -1)
        curses.init_pair(_YELLOW, curses.COLOR_YELLOW, -1)
        curses.init_pair(_RED,    curses.COLOR_RED,    -1)
        curses.init_pair(_CYAN,   curses.COLOR_CYAN,   -1)
        curses.init_pair(_WHITE,  curses.COLOR_WHITE,  -1)
        curses.init_pair(_GREY,   8 if curses.COLORS >= 256 else curses.COLOR_WHITE, -1)
        stdscr.nodelay(True)   # non-blocking getch

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._sub = self.create_subscription(
            String, TOPIC, self._path_cb, qos
        )

        # --- Stats state ---
        self._fps_ema        = 0.0
        self._bw_ema         = 0.0   # bytes/s (file size / interval)
        self._frame_count    = 0
        self._last_recv_time = None
        self._run_dir        = ''

        # Rolling window (last 60 frames) for jitter / stddev
        self._intervals: collections.deque = collections.deque(maxlen=60)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        self.get_logger().info(f'Subscribed to {TOPIC}')

    # ------------------------------------------------------------------
    def _path_cb(self, msg: String):
        # Message format: "path|width|height"
        parts = msg.data.split('|')
        path  = parts[0]
        img_w = int(parts[1]) if len(parts) >= 3 else 0
        img_h = int(parts[2]) if len(parts) >= 3 else 0
        now   = time.monotonic()

        # --- Timing ---
        if self._last_recv_time is not None:
            dt = now - self._last_recv_time
            if dt > 0:
                inst_fps = 1.0 / dt
                try:
                    inst_bw = os.path.getsize(path) / dt
                except OSError:
                    inst_bw = 0.0
                if self._fps_ema == 0.0:
                    self._fps_ema = inst_fps
                    self._bw_ema  = inst_bw
                else:
                    self._fps_ema = STATS_ALPHA * inst_fps + (1 - STATS_ALPHA) * self._fps_ema
                    self._bw_ema  = STATS_ALPHA * inst_bw  + (1 - STATS_ALPHA) * self._bw_ema
                self._intervals.append(dt)
        self._last_recv_time = now
        self._run_dir = os.path.dirname(path)
        self._frame_count += 1

        # --- Load from disk ---
        if path.endswith('.bin'):
            if img_w == 0 or img_h == 0:
                self.get_logger().warn(f'Missing shape metadata for {path}')
                return
            import numpy as np
            try:
                frame = np.frombuffer(open(path, 'rb').read(), dtype=np.uint8).reshape(img_h, img_w, 3)
            except Exception as ex:
                self.get_logger().warn(f'Could not read {path}: {ex}')
                return
        else:
            frame = cv2.imread(path)
            if frame is None:
                self.get_logger().warn(f'Could not read {path}')
                return
        img_h, img_w = frame.shape[:2]

        # --- Stats in terminal ---
        self._print_stats(path, img_w, img_h)

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # q or Esc
            self.get_logger().info('Quit requested.')
            raise SystemExit

        # Also quit on 'q' in the terminal
        ch = self._scr.getch()
        if ch in (ord('q'), ord('Q'), 27):
            raise SystemExit

    # ------------------------------------------------------------------
    def _print_stats(self, path: str, orig_w: int, orig_h: int):
        jitter_ms = 0.0
        if len(self._intervals) > 1:
            import numpy as np
            jitter_ms = float(np.std(list(self._intervals))) * 1000.0

        bw_mbps = self._bw_ema * 8 / 1e6

        scr = self._scr
        scr.erase()
        max_y, max_x = scr.getmaxyx()

        def addrow(row, label, value, pair):
            if row >= max_y:
                return
            scr.addstr(row, 0,  f'  {label:<10}', curses.color_pair(_GREY))
            scr.addstr(row, 14, value[:max_x - 15], curses.color_pair(pair))

        title = ' FLIR Camera Stats  (q = quit) '
        scr.addstr(0, 0, title.center(min(max_x - 1, 60), '─'),
                   curses.color_pair(_CYAN) | curses.A_BOLD)

        if self._fps_ema >= 25:
            fps_pair = _GREEN
        elif self._fps_ema >= 15:
            fps_pair = _YELLOW
        else:
            fps_pair = _RED

        addrow(2,  'Topic',   TOPIC,                                       _WHITE)
        addrow(3,  'Run dir', self._run_dir,                               _WHITE)
        addrow(4,  'File',    os.path.basename(path),                      _WHITE)
        addrow(5,  'Res',     f'{orig_w} x {orig_h}', _WHITE)
        addrow(6,  'FPS',     f'{self._fps_ema:.2f} fps',                  fps_pair)
        addrow(7,  'BW',      f'{bw_mbps:.2f} Mbps  ({self._bw_ema/1e6:.2f} MB/s)', _CYAN)
        addrow(8,  'Jitter',  f'{jitter_ms:.2f} ms  (stddev)',             _YELLOW)
        addrow(9,  'Frames',  str(self._frame_count),                      _WHITE)

        scr.refresh()


def main(args=None):
    def _run(stdscr):
        rclpy.init(args=args)
        node = FLIRViewer(stdscr)
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            cv2.destroyAllWindows()
            node.destroy_node()
            rclpy.shutdown()

    curses.wrapper(_run)


if __name__ == '__main__':
    main()
