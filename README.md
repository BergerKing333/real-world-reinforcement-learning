# Catheter Data Collection System

ROS2 Jazzy pipeline for automated catheter manipulation and FLIR GigE camera capture.

---

## Quick Start

### 1. Environment setup

```bash
# Camera / ROS2 nodes (PySpin, rclpy, pyserial, opencv, numpy)
mamba activate camera

# Dataset visualizer (Streamlit + Plotly)
mamba activate viz
```

### 2. Launch the system (3 terminals)

```bash
# Terminal 1 — Camera node
mamba activate camera
python3 ./camera/FLIR.py

# Terminal 2 — Catheter driver
mamba activate camera
python3 ./driver.py

# Terminal 3 — Data collection
mamba activate camera
python3 ./collect.py --trials 10 --steps 50 --output /tmp/catheter_dataset
```

### 3. Live viewer (optional, 4th terminal)

```bash
mamba activate camera
python3 ./camera/FLIR_Viewer.py
```

### 4. Browse the dataset

```bash
mamba activate viz
streamlit run visualizer.py -- /tmp/catheter_dataset
# Opens at http://localhost:8501
```

---

## Project Structure

```
Camera/
├── camera/
│   ├── FLIR.py            # FLIR GigE camera ROS2 node
│   └── FLIR_Viewer.py     # Live camera viewer (OpenCV + curses stats)
├── driver.py              # Catheter serial driver ROS2 node
├── collect.py             # Data collection pipeline
├── visualizer.py          # Streamlit dataset browser
└── README.md
```

---

## Components

### camera/FLIR.py — Camera Node

Grabs frames from a FLIR GigE camera at `169.255.0.6`. Two modes:

- **On-demand** (default) — holds the latest frame in memory. Send a folder path to `/flir/save_request` to save the current frame as a raw BGR8 `.bin` file. The path is published to `/flir/save_response` immediately; the disk write happens in a background thread.
- **`--save-all`** — saves every frame to `/tmp/flir/<timestamp>/` continuously.

```bash
python3 ./camera/FLIR.py              # on-demand mode
python3 ./camera/FLIR.py --save-all   # continuous save
```

**ROS2 Topics**

| Topic | Type | Dir | Description |
|---|---|---|---|
| `/flir/image_path` | `String` | pub | `path\|W\|H` of last saved frame |
| `/flir/save_request` | `String` | sub | Folder path to trigger a save (on-demand only) |
| `/flir/save_response` | `String` | pub | `path\|W\|H` response (on-demand only) |

**Configuration** (constants at top of file)

| Constant | Default | Notes |
|---|---|---|
| `CAMERA_IP` | `169.255.0.6` | GigE camera address |
| `EXPOSURE_US` | `20000.0` | Exposure time µs. Max fps ≈ 1/exposure_s |
| `GAIN_DB` | `0.0` | Analogue gain dB |
| `PUBLISH_HZ` | `60` | Target frame rate — clamped to camera max |
| `WRITE_THREADS` | `4` | Background disk-write workers |

**Subnet recovery** — If SpinView left the camera on a mismatched subnet (error `-1015`), the node automatically issues a GevCP `ForceIP` to reassert the IP. No manual fix needed.

---

### camera/FLIR_Viewer.py — Live Viewer

Subscribes to `/flir/image_path`, loads raw `.bin` frames from disk, and displays them in an OpenCV window. A curses stats panel in the terminal shows FPS, bandwidth, jitter, and frame count.

```bash
python3 ./camera/FLIR_Viewer.py
```

Press **q** or **Esc** in either the image window or the terminal to quit.

**Stats panel fields**: Topic, Run dir, File, Resolution, FPS (colour-coded: green ≥ 25, yellow ≥ 15, red < 15), Bandwidth (Mbps), Jitter (stddev ms), Frame count.

---

### driver.py — Catheter Driver

Bridges a Trust serial catheter controller (`/dev/ttyUSB0`, 115200 baud) to ROS2 topics.

```bash
python3 ./driver.py                          # default port
python3 ./driver.py --port /dev/ttyUSB1      # alternate port
```

**ROS2 Topics**

| Topic | Type | Dir | Description |
|---|---|---|---|
| `/catheter/command` | `String` | sub | JSON command (see below) |
| `/catheter/state` | `String` | pub | JSON state at 10 Hz |
| `/catheter/feedback` | `String` | pub | Raw serial response lines |
| `/catheter/position_reached` | `Bool` | pub | `true` when motion completes |

**Command JSON**

```json
{"insertion": 0.5, "rotation": 1.047, "relative": true}
```

- `insertion` — centimetres (500 units = 1 cm)
- `rotation` — radians (2650 units = 2π rad)
- `relative` — `true` = delta from current, `false` = absolute position
- `{"reset": true}` — return to home (0, 0)

Motion timeout: 30 s. If `POSITION_REACHED` is not received within the timeout the driver declares the motion complete anyway.

---

### collect.py — Data Collection

Orchestrates random catheter movements paired with image captures. Resets the catheter to home at the start of each trial, then performs N random steps, clamping to safety limits.

```bash
python3 ./collect.py --trials 10 --steps 50 --output /tmp/catheter_dataset --delay 0
```

| Flag | Default | Description |
|---|---|---|
| `--trials` | `10` | Number of trials (resets between each) |
| `--steps` | `100` | Random actions per trial |
| `--output` | `/tmp/catheter_dataset` | Output directory |
| `--delay` | `0.0` | Extra sleep (s) between steps |

**Safety limits**

| Limit | Value |
|---|---|
| Relative insertion per step | −0.2 … +0.4 cm |
| Relative rotation per step | ±π/6 rad |
| Absolute insertion | 0 … 5 cm |
| Absolute rotation | ±2π rad |

**Dataset format** — `<output>/dataset.jsonl`, one JSON object per line:

```json
{
  "step": 0,
  "trial": 0,
  "trial_id": "trial_0000_step_000",
  "timestamp_iso": "2026-03-25T17:16:45.087904+00:00",
  "image_path": "/tmp/catheter_dataset/images/350000406320968.bin",
  "image_w": 2448,
  "image_h": 2048,
  "state_before": { "insertion_cm": 0.0, "rotation_rad": 0.0, "insertion_units": 0.0, "rotation_units": 0.0 },
  "command": { "insertion": 0.0, "rotation": 0.4529, "relative": true },
  "state_after": { "insertion_cm": 0.0, "rotation_rad": 0.4529, "insertion_units": 0.0, "rotation_units": 191.0 }
}
```

Images are raw BGR8: `H × W × 3` bytes, `uint8`. Reload:

```python
import numpy as np
frame = np.fromfile("path.bin", dtype=np.uint8).reshape(2048, 2448, 3)
```

---

### visualizer.py — Dataset Browser

Interactive Streamlit app for exploring collected datasets.

```bash
mamba activate viz
streamlit run visualizer.py -- /tmp/catheter_dataset
```

**Features**:
- Trial selector + step slider to scrub through frames
- Camera image display (raw `.bin` loaded on-the-fly)
- Insertion & rotation trajectory plots with current-step marker
- Polar view (insertion = radius, rotation = angle)
- Per-step command bar charts
- State before/after metric cards
- Raw JSON record inspector

**Environment**: `viz` (Python 3.12) — `streamlit`, `plotly`, `pandas`, `numpy`, `pillow`.

---

## Environments

| Name | Purpose | Key packages |
|---|---|---|
| `camera` | ROS2 nodes | `rclpy`, `PySpin`, `opencv-python`, `pyserial`, `numpy` |
| `viz` | Visualizer | `streamlit`, `plotly`, `pandas`, `numpy`, `pillow` |

```bash
# Create viz env from scratch
mamba create -n viz python=3.12 -y
mamba run -n viz pip install streamlit plotly pandas numpy pillow
```

---

## ROS2 Topic Map

```
/flir/save_request  ──→  [FLIR.py]  ──→  /flir/save_response
                                     ──→  /flir/image_path  ──→  [FLIR_Viewer.py]

/catheter/command   ──→  [driver.py] ──→  /catheter/state
                                     ──→  /catheter/feedback
                                     ──→  /catheter/position_reached

[collect.py]  ←──→  /flir/save_request, /flir/save_response
              ←──→  /catheter/command, /catheter/state, /catheter/position_reached
```
