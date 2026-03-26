"""
guidewire_with_goals.py

Single-script pipeline:
  1. Segment the guidewire (v3: multi-angle top-hat, top-percentile threshold)
  2. Skeletonise → find the free tip
  3. Detect dots with Hough circles (only used for safe-zone masking)
  4. Build a safe-zone mask (valid region ∩ not-dot ∩ not-wire ∩ annular ring)
  5. Sample N random goal positions from the safe zone
  6. Visualise everything

Usage
-----
python guidewire_with_goals.py image.png [top_percent] [n_goals] [max_dist]

Examples
--------
python guidewire_with_goals.py frame.png                  # all defaults
python guidewire_with_goals.py frame.png 0.5 5 130        # explicit args
"""

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.morphology import skeletonize


# ════════════════════════════════════════════════════════
#  PART 1 – GUIDEWIRE SEGMENTATION  (v3, unchanged)
# ════════════════════════════════════════════════════════

def multi_angle_tophat(gray: np.ndarray, ksize: int = 23, n_angles: int = 6) -> np.ndarray:
    """Black top-hat with line kernels at n_angles orientations; returns max response."""
    best = np.zeros_like(gray, dtype=np.float32)
    for i in range(n_angles):
        angle  = i * 180.0 / n_angles
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        kernel[ksize // 2, :] = 1                              # horizontal line
        M      = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1.0)
        rotated = cv2.warpAffine(kernel, M, (ksize, ksize),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT)
        th = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rotated)
        best = np.maximum(best, th.astype(np.float32))
    return best.astype(np.uint8)


def keep_longest_skeleton_component(binary: np.ndarray) -> np.ndarray:
    """
    For each connected component in `binary`, skeletonise it individually
    and count skeleton pixels.  Return a mask containing only the component
    whose skeleton is longest — i.e. the wire, not noise or thick frame edges.

    This beats filtering by blob area because:
      - Frame edges are thick → huge area but SHORT skeleton relative to size.
      - Wire is thin and long → modest area but LONG skeleton.
      - Noise blobs are small → short skeleton, filtered out.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1:
        return binary   # nothing to filter

    best_label, best_skel_len = -1, 0
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < 30:
            continue    # ignore tiny specks
        component = np.uint8(labels == lbl) * 255
        skel_len  = int(np.sum(skeletonize(component > 0)))
        print(f"    component {lbl}: area={stats[lbl, cv2.CC_STAT_AREA]}  skel_len={skel_len}")
        if skel_len > best_skel_len:
            best_skel_len = skel_len
            best_label    = lbl

    if best_label == -1:
        return np.zeros_like(binary)

    print(f"  [filter] keeping component {best_label} (skel_len={best_skel_len})")
    return np.uint8(labels == best_label) * 255


def segment_wire(gray: np.ndarray, top_percent: float = 0.5):
    """
    v3 segmentation: top-hat → top-percentile threshold → longest-skeleton
    component → close gaps → skeleton.

    Returns
    -------
    tophat     : raw top-hat response image
    wire_mask  : filled binary wire mask (single component: the guidewire)
    skeleton   : 1-px centre-line
    """
    gray_blur  = cv2.GaussianBlur(gray, (3, 3), 0)
    tophat     = multi_angle_tophat(gray_blur, ksize=23, n_angles=6)

    thresh_val = np.percentile(tophat, 100.0 - top_percent)
    _, binary  = cv2.threshold(tophat, thresh_val, 255, cv2.THRESH_BINARY)
    binary     = binary.astype(np.uint8)

    # Keep only the component with the longest skeleton (= the wire)
    wire_only = keep_longest_skeleton_component(binary)

    # Close small gaps along the wire
    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wire_mask = cv2.morphologyEx(wire_only, cv2.MORPH_CLOSE, k, iterations=3)

    skeleton  = np.uint8(skeletonize(wire_mask > 0)) * 255

    print(f"  [segment] threshold={thresh_val:.1f}  (top {top_percent}% of pixels)")
    return tophat, wire_mask, skeleton


# ════════════════════════════════════════════════════════
#  PART 2 – DOT DETECTION  (Hough, for safe-zone only)
# ════════════════════════════════════════════════════════

def detect_dots(gray: np.ndarray,
                min_r: int = 15, max_r: int = 45) -> np.ndarray:
    """
    Detect the regular dot grid with Hough circles.
    Returns a filled binary mask (255 = dot interior).
    Only used for safe-zone computation, NOT for wire segmentation.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.5, minDist=30,
        param1=60, param2=25,
        minRadius=min_r, maxRadius=max_r,
    )
    dot_mask = np.zeros_like(gray, dtype=np.uint8)
    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            cv2.circle(dot_mask, (cx, cy), r, 255, -1)
    print(f"  [dots] detected {0 if circles is None else len(circles[0])} circles")
    return dot_mask


# ════════════════════════════════════════════════════════
#  PART 3 – TIP DETECTION
# ════════════════════════════════════════════════════════

def trace_branch_length(skeleton: np.ndarray,
                         start_rc: tuple[int, int],
                         max_steps: int = 2000) -> int:
    """
    Walk along the skeleton from start_rc until hitting a junction or dead end.
    Returns the number of steps taken (= branch length in pixels).
    A stray noise pixel has branch length 1; a real wire tip has a long branch.
    """
    skel = skeleton > 0
    visited = set()
    r, c = start_rc
    visited.add((r, c))

    for step in range(max_steps):
        # 8-connected neighbours on the skeleton not yet visited
        neighbours = [
            (r + dr, c + dc)
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
            and 0 <= r + dr < skel.shape[0]
            and 0 <= c + dc < skel.shape[1]
            and skel[r + dr, c + dc]
            and (r + dr, c + dc) not in visited
        ]
        if len(neighbours) == 0:
            break           # dead end (another endpoint or isolated pixel)
        if len(neighbours) > 1:
            break           # junction — stop here, this is a branch point
        r, c = neighbours[0]
        visited.add((r, c))

    return len(visited)


def find_wire_tip(skeleton: np.ndarray,
                  anchor_side: str = "left",
                  min_branch_len: int = 20) -> tuple[int, int]:
    """
    Find the free tip of the wire skeleton.

    Steps:
      1. Find all endpoints (pixels with exactly 1 skeleton neighbour).
      2. Trace the branch length from each endpoint.
      3. Discard endpoints whose branch is shorter than min_branch_len
         — these are stray pixels / tiny noise fragments.
      4. Among the remaining valid endpoints, pick the one furthest
         from the anchor side (= the free tip, not the device end).

    min_branch_len: minimum pixels in the branch for an endpoint to be
                    considered a real tip.  20px works well; raise if
                    short noise branches still sneak through.
    """
    kernel          = np.ones((3, 3), dtype=np.uint8)
    neighbour_count = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel)
    endpoints       = np.argwhere((skeleton > 0) & (neighbour_count == 2))

    if len(endpoints) == 0:
        raise RuntimeError("No skeleton endpoints found – wire mask may be empty.")

    # Filter by branch length
    valid = []
    for ep in endpoints:
        blen = trace_branch_length(skeleton, tuple(ep))
        print(f"    endpoint ({ep[0]}, {ep[1]}): branch_len={blen}")
        if blen >= min_branch_len:
            valid.append((blen, tuple(ep)))

    if not valid:
        raise RuntimeError(
            f"No endpoints with branch_len >= {min_branch_len}. "
            "Try lowering min_branch_len or check the skeleton."
        )

    # Among valid endpoints, pick furthest from anchor
    eps_arr = np.array([ep for _, ep in valid])
    if anchor_side == "left":
        idx = np.argmax(eps_arr[:, 1])    # largest col = furthest right
    elif anchor_side == "right":
        idx = np.argmin(eps_arr[:, 1])
    elif anchor_side == "top":
        idx = np.argmax(eps_arr[:, 0])
    else:
        idx = np.argmin(eps_arr[:, 0])

    tip = valid[idx][1]
    print(f"  [tip] row={tip[0]}, col={tip[1]}  (branch_len={valid[idx][0]})")
    return tip


# ════════════════════════════════════════════════════════
#  PART 4 – SAFE-ZONE MASK
# ════════════════════════════════════════════════════════

def build_safe_mask(gray       : np.ndarray,
                    dot_mask   : np.ndarray,
                    wire_mask  : np.ndarray,
                    tip_rc     : tuple[int, int],
                    min_dist   : int = 20,
                    max_dist   : int = 130,
                    dot_margin : int = 8,
                    wire_margin: int = 10) -> np.ndarray:
    """
    A pixel is safe iff:
      • It is inside the image (shrunk by a small fixed border)
      • It is NOT inside a dot  (+ dot_margin px buffer)
      • It is NOT on the wire   (+ wire_margin px buffer)
      • Its distance from the tip is in [min_dist, max_dist]
    """
    h, w = gray.shape

    # Shrink away from image edges (but don't mask the left side aggressively
    # since the wire enters from there – a small 5px strip is enough)
    valid = np.zeros_like(gray, dtype=np.uint8)
    valid[10:h-10, 5:w-10] = 255
    # Also exclude very dark (off-screen) regions
    bright = np.uint8(gray > 15) * 255
    k_big  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    valid  = cv2.bitwise_and(valid, cv2.erode(bright, k_big))

    # Exclusion zones
    k_dot  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2*dot_margin+1,  2*dot_margin+1))
    k_wire = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2*wire_margin+1, 2*wire_margin+1))
    no_dot  = cv2.dilate(dot_mask,  k_dot)
    no_wire = cv2.dilate(wire_mask, k_wire)

    # Annular ring around the tip
    tr, tc = tip_rc
    ys, xs = np.ogrid[:h, :w]
    dist   = np.sqrt((ys - tr)**2 + (xs - tc)**2)
    ring   = np.uint8((dist >= min_dist) & (dist <= max_dist)) * 255

    safe = cv2.bitwise_and(valid, ring)
    safe[no_dot  > 0] = 0
    safe[no_wire > 0] = 0

    n_safe = int(np.sum(safe > 0))
    print(f"  [safe zone] {n_safe} candidate pixels")
    return safe


# ════════════════════════════════════════════════════════
#  PART 5 – GOAL SAMPLING
# ════════════════════════════════════════════════════════

def sample_goals(safe_mask  : np.ndarray,
                 n          : int = 5,
                 min_spacing: int = 20,
                 rng        : np.random.Generator | None = None) -> list[tuple[int, int]]:
    """
    Uniformly sample n goals from safe_mask, enforcing min_spacing between them.
    Returns list of (row, col).
    """
    if rng is None:
        rng = np.random.default_rng()

    candidates = np.argwhere(safe_mask > 0)
    if len(candidates) == 0:
        raise RuntimeError("Safe mask is empty. Try increasing max_dist or reducing margins.")

    rng.shuffle(candidates)
    goals = []
    for rc in candidates:
        rc = tuple(rc)
        if all(np.linalg.norm(np.array(rc) - np.array(g)) >= min_spacing for g in goals):
            goals.append(rc)
        if len(goals) == n:
            break

    if len(goals) < n:
        print(f"  [goals] warning: only {len(goals)}/{n} goals fit with min_spacing={min_spacing}px")
    return goals


# ════════════════════════════════════════════════════════
#  VISUALISATION
# ════════════════════════════════════════════════════════

def visualise_all(gray      : np.ndarray,
                  tophat    : np.ndarray,
                  wire_mask : np.ndarray,
                  skeleton  : np.ndarray,
                  dot_mask  : np.ndarray,
                  safe_mask : np.ndarray,
                  tip_rc    : tuple[int, int],
                  goals     : list[tuple[int, int]],
                  min_dist  : int,
                  max_dist  : int,
                  save_path : str = "guidewire_goals.png"):

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Guidewire Segmentation + Goal Sampling", fontsize=14)

    tr, tc = tip_rc

    def add_goals(ax):
        ax.plot(tc, tr, "r*", markersize=14, zorder=5, label="tip")
        for i, (gr, gc) in enumerate(goals):
            ax.plot(gc, gr, "o", color="lime", markersize=11,
                    markeredgecolor="black", markeredgewidth=1.5, zorder=5)
            ax.text(gc + 6, gr - 6, str(i+1), color="lime",
                    fontsize=9, fontweight="bold", zorder=6)

    # 1. Grayscale
    axes[0,0].imshow(gray, cmap="gray")
    axes[0,0].set_title("1. Grayscale"); axes[0,0].axis("off")

    # 2. Top-hat response
    axes[0,1].imshow(tophat, cmap="gray")
    axes[0,1].set_title("2. Top-Hat Response"); axes[0,1].axis("off")

    # 3. Wire mask + skeleton
    axes[0,2].imshow(gray, cmap="gray")
    skel_rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
    skel_rgb[wire_mask > 0] = (255, 140,   0)
    skel_rgb[skeleton  > 0] = (255,   0,   0)
    axes[0,2].imshow(skel_rgb, alpha=0.7)
    axes[0,2].plot(tc, tr, "r*", markersize=14)
    axes[0,2].set_title("3. Wire Mask + Skeleton + Tip"); axes[0,2].axis("off")

    # 4. Detected dots
    axes[1,0].imshow(gray, cmap="gray")
    dot_rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
    dot_rgb[dot_mask > 0] = (80, 120, 255)
    axes[1,0].imshow(dot_rgb, alpha=0.5)
    axes[1,0].set_title("4. Detected Dots (for safe zone)"); axes[1,0].axis("off")

    # 5. Safe zone
    axes[1,1].imshow(gray, cmap="gray", alpha=0.6)
    safe_rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
    safe_rgb[safe_mask > 0] = (0, 220, 100)
    axes[1,1].imshow(safe_rgb, alpha=0.5)
    for r, ls in [(min_dist, "--"), (max_dist, "-")]:
        axes[1,1].add_patch(Circle((tc, tr), r, fill=False,
                                   edgecolor="yellow", lw=1.5, linestyle=ls))
    add_goals(axes[1,1])
    axes[1,1].legend(loc="upper right", fontsize=8)
    axes[1,1].set_title("5. Safe Zone + Goals"); axes[1,1].axis("off")

    # 6. Final overlay
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).copy()
    overlay[wire_mask > 0] = (255, 140,  0)
    overlay[skeleton  > 0] = (255,   0,  0)
    axes[1,2].imshow(overlay)
    axes[1,2].add_patch(Circle((tc, tr), max_dist, fill=False,
                               edgecolor="yellow", lw=1.5))
    add_goals(axes[1,2])
    axes[1,2].legend(loc="upper right", fontsize=8)
    axes[1,2].set_title("6. Final Overlay"); axes[1,2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [vis] saved → {save_path}")


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════

def run(image_path : str,
        top_percent: float = 0.5,
        n_goals    : int   = 5,
        min_dist   : int   = 20,
        max_dist   : int   = 130,
        anchor_side: str   = "left",
        seed       : int | None = None):

    print("=== Guidewire + Goal Pipeline ===")

    # Load
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Stage 1 – segment wire
    print("[1] Segmenting wire...")
    tophat, wire_mask, skeleton = segment_wire(gray, top_percent=top_percent)

    # Stage 2 – find tip
    print("[2] Finding tip...")
    tip_rc = find_wire_tip(skeleton, anchor_side=anchor_side)

    # Stage 3 – detect dots (safe-zone use only)
    print("[3] Detecting dots...")
    dot_mask = detect_dots(gray)

    # Stage 4 – safe zone
    print("[4] Building safe zone...")
    safe_mask = build_safe_mask(gray, dot_mask, wire_mask, tip_rc,
                                min_dist=min_dist, max_dist=max_dist)

    # Stage 5 – sample goals
    print("[5] Sampling goals...")
    rng   = np.random.default_rng(seed)
    goals = sample_goals(safe_mask, n=n_goals, rng=rng)
    for i, (r, c) in enumerate(goals):
        print(f"       Goal {i+1}: (row={r}, col={c})")

    # Stage 6 – visualise
    print("[6] Visualising...")
    visualise_all(gray, tophat, wire_mask, skeleton, dot_mask,
                  safe_mask, tip_rc, goals, min_dist, max_dist)

    return tip_rc, goals


if __name__ == "__main__":
    path        = sys.argv[1] if len(sys.argv) > 1 else "image.png"
    top_percent = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    n_goals     = int(sys.argv[3])   if len(sys.argv) > 3 else 5
    max_dist    = int(sys.argv[4])   if len(sys.argv) > 4 else 40

    run(path, top_percent=top_percent, n_goals=n_goals, max_dist=max_dist)