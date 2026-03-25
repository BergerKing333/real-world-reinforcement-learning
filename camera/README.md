# FLIR Camera — ROS2 Jazzy

GigE Vision camera node for the FLIR Blackfly / Grasshopper series at `169.255.0.6`.

## Files

| File | Purpose |
|---|---|
| `FLIR.py` | Camera node — grabs frames, saves on request or continuously |
| `FLIR_Viewer.py` | Viewer — subscribes to saved-file paths, displays frames live |

---

## FLIR.py

### Dependencies

```
PySpin  cv2  rclpy  std_msgs
```

### Configuration (top of file)

| Constant | Default | Description |
|---|---|---|
| `CAMERA_IP` | `169.255.0.6` | GigE camera IP |
| `EXPOSURE_US` | `20000.0` | Exposure time in microseconds |
| `GAIN_DB` | `0.0` | Analogue gain in dB |
| `GAMMA` | `1.0` | Gamma (1.0 = off) |
| `PUBLISH_HZ` | `60` | Target frame rate — clamped to camera maximum |
| `WRITE_THREADS` | `4` | Background worker threads for disk writes |

> **Frame rate vs exposure:** `max_fps ≈ 1 / exposure_s`. At 20 ms exposure the ceiling is ~22 fps. Lower `EXPOSURE_US` to unlock higher rates.

### Usage

```bash
# On-demand mode (default) — frames held in memory, saved only when requested
python3 ./camera/FLIR.py

# Save every frame continuously to /tmp/flir/<YYYYMMDD_HHMMSS>/
python3 ./camera/FLIR.py --save-all
```

### ROS2 Topics

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/flir/image_path` | `std_msgs/String` | publish | Path of the most recently saved file, format `path\|W\|H` |
| `/flir/save_request` | `std_msgs/String` | subscribe | *(on-demand mode only)* Send a folder path to trigger a save |
| `/flir/save_response` | `std_msgs/String` | publish | *(on-demand mode only)* Responds with `path\|W\|H` immediately (before write finishes) |

### On-demand save

Publish the destination folder to `/flir/save_request`. The node responds immediately on `/flir/save_response` with the full file path (the disk write happens in the background so the caller is not blocked). The file is also echoed on `/flir/image_path` so the viewer picks it up automatically.

```bash
ros2 topic pub --once /flir/save_request std_msgs/String "data: '/tmp/my_captures'"
```

The saved file is a raw BGR8 binary: `height × width × 3` bytes, `uint8`. Reload with:

```python
import numpy as np, cv2
frame = np.frombuffer(open('path.bin', 'rb').read(), dtype=np.uint8).reshape(H, W, 3)
cv2.imshow('frame', frame); cv2.waitKey(0)
```

### Subnet / IP recovery

If SpinView left the camera on a mismatched subnet the node detects error `[-1015]` and automatically issues a GevCP `ForceIP` command to reassert `169.255.0.6`, waits for re-enumeration, and retries `Init()` — no manual intervention needed.

---

## FLIR_Viewer.py

Live viewer that reads saved files from disk and displays them in an OpenCV window, with a stats panel in the terminal.

### Dependencies

```
cv2  numpy  rclpy  std_msgs  curses (stdlib)
```

### Usage

```bash
python3 ./camera/FLIR_Viewer.py
```

Run alongside `FLIR.py` in a separate terminal. Press **q** or **Esc** (in either the image window or the terminal) to quit.

### ROS2 Topics

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/flir/image_path` | `std_msgs/String` | subscribe | Receives `path\|W\|H` for each saved frame |

### Terminal stats panel

| Field | Description |
|---|---|
| Topic | Subscribed topic |
| Run dir | Parent directory of the current run |
| File | Current filename |
| Res | Full image resolution |
| FPS | EMA-smoothed receive rate — green ≥ 25, yellow ≥ 15, red < 15 |
| BW | Inferred throughput based on file size / interval |
| Jitter | Stddev of inter-frame intervals (last 60 frames) |
| Frames | Total frames received since launch |
