"""
Guidewire extraction and simple tip detection.

This version keeps the original segmentation pipeline, but also returns a
single tip point in image coordinates so the RL environment can work in
pixel space.

Current tip heuristic:
- skeletonise the detected guidewire
- find skeleton endpoints
- choose the endpoint farthest to the right

That assumption matches our current setup where the guidewire enters from
the left side of the image. It is simple, but good enough as a first draft.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


def multi_angle_tophat(gray: np.ndarray, ksize: int = 23, n_angles: int = 6) -> np.ndarray:
    best = np.zeros_like(gray, dtype=np.float32)

    for i in range(n_angles):
        angle = i * 180.0 / n_angles

        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        kernel[ksize // 2, :] = 1

        M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            kernel,
            M,
            (ksize, ksize),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )

        th = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rotated)
        best = np.maximum(best, th.astype(np.float32))

    return best.astype(np.uint8)


def find_skeleton_endpoints(skeleton: np.ndarray):
    """
    Return all endpoints in a 1-pixel skeleton.

    An endpoint is a skeleton pixel with exactly one 8-connected neighbour.
    """
    sk = (skeleton > 0).astype(np.uint8)
    h, w = sk.shape
    endpoints = []

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if sk[y, x] == 0:
                continue

            patch = sk[y - 1:y + 2, x - 1:x + 2]
            neighbour_count = int(np.sum(patch)) - 1  # exclude the centre pixel

            if neighbour_count == 1:
                endpoints.append((x, y))

    return endpoints


def choose_tip_from_endpoints(endpoints):
    """
    Pick the most likely guidewire tip.

    For now I am using a simple rule:
    choose the endpoint with the largest x-coordinate.
    """
    if len(endpoints) == 0:
        return None

    return max(endpoints, key=lambda p: p[0])


def extract_tip_from_image(image_bgr: np.ndarray, top_percent: float = 0.5):
    """
    Process an already-loaded BGR image and return:
    - wire mask
    - skeleton
    - tip_xy

    tip_xy is returned as (x, y), or None if no valid tip is found.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Strong dark-line enhancement across several angles
    tophat = multi_angle_tophat(gray, ksize=23, n_angles=6)

    # Keep only the strongest responses
    thresh_val = np.percentile(tophat, 100 - top_percent)
    _, binary = cv2.threshold(tophat, thresh_val, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    clean = np.zeros_like(binary)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 30:
            clean[labels == lbl] = 255

    # Close small gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wire_mask = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=3)

    # 1-pixel centreline
    skeleton = np.uint8(skeletonize(wire_mask > 0)) * 255

    # Endpoints and tip selection
    endpoints = find_skeleton_endpoints(skeleton)
    tip_xy = choose_tip_from_endpoints(endpoints)

    return wire_mask, skeleton, tip_xy


def extract_guidewire_and_tip(
    image_path: str,
    output_path: str = "guidewire_result_v4.png",
    top_percent: float = 0.5,
    show_plot: bool = True,
):
    """
    Load an image from disk, run guidewire extraction, mark the detected tip,
    and optionally plot the intermediate stages.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    wire_mask, skeleton, tip_xy = extract_tip_from_image(img, top_percent=top_percent)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    overlay[wire_mask > 0] = (0, 180, 255)   # orange mask
    overlay[skeleton > 0] = (0, 0, 255)      # red skeleton

    endpoints = find_skeleton_endpoints(skeleton)
    for (x, y) in endpoints:
        cv2.circle(overlay, (x, y), 4, (255, 255, 0), -1)

    if tip_xy is not None:
        cv2.circle(overlay, tip_xy, 8, (0, 255, 0), -1)
        cv2.putText(
            overlay,
            "TIP",
            (tip_xy[0] + 10, tip_xy[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_path, overlay)
    print(f"Saved -> {output_path}")
    print(f"Endpoints found: {len(endpoints)}")
    print(f"Tip: {tip_xy}")

    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Guidewire extraction + tip detection", fontsize=13)

        panels = [
            (gray, "1. Grayscale"),
            (wire_mask, "2. Wire mask"),
            (skeleton, "3. Skeleton"),
            (cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), "4. Overlay + tip"),
            (wire_mask, "5. Mask"),
            (skeleton, "6. Skeleton"),
        ]

        for ax, (im, title) in zip(axes.flat, panels):
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig("guidewire_pipeline_v4.png", dpi=150, bbox_inches="tight")
        plt.show()

    return wire_mask, skeleton, tip_xy


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "image.png"
    top_percent = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    extract_guidewire_and_tip(
        image_path=path,
        top_percent=top_percent,
        show_plot=True,
    )