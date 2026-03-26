"""
Guidewire extraction v3 – simple brightest-pixel threshold.

The guidewire is the brightest response in the top-hat image.
Just keep the top N% of pixels and clean up noise.
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
        rotated = cv2.warpAffine(kernel, M, (ksize, ksize),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT)
        th = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rotated)
        best = np.maximum(best, th.astype(np.float32))
    return best.astype(np.uint8)


def extract_guidewire(image_path: str,
                      output_path: str = "guidewire_result_v3.png",
                      top_percent: float = 0.2):   # ← only knob you need
    """
    top_percent: keep only the brightest X% of top-hat pixels.
                 0.5 = top 0.5 %  →  very strict (fewer false positives)
                 1.5 = top 1.5 %  →  more lenient  (catches faint wire sections)
    """

    # 1. Load & grayscale
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. Multi-angle black top hat
    tophat = multi_angle_tophat(gray, ksize=23, n_angles=6)

    # 3. Threshold at the (100 - top_percent) percentile
    #    e.g. top_percent=0.5  →  thresh = 99.5th percentile value
    thresh_val = np.percentile(tophat, 100 - top_percent)
    _, binary = cv2.threshold(tophat, thresh_val, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    # 4. Remove tiny noise blobs (keep components >= 30 px)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    clean = np.zeros_like(binary)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 30:
            clean[labels == lbl] = 255

    # 5. Close small gaps along the wire
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wire_mask = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=3)

    # 6. Skeletonise → 1-px centre-line
    skeleton = np.uint8(skeletonize(wire_mask > 0)) * 255

    # 7. Overlay
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[wire_mask > 0] = (0, 180, 255)  # orange fill
    overlay[skeleton  > 0] = (0,   0, 255)  # red centre-line
    cv2.imwrite(output_path, overlay)
    print(f"Threshold value used: {thresh_val:.1f}  (top {top_percent}% of pixels)")
    print(f"Saved → {output_path}")

    # 8. Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Guidewire v3 - Top {top_percent}% brightest top-hat pixels", fontsize=13)
    panels = [
        (gray,      "1. Grayscale"),
        (tophat,    f"2. Top Hat  (thresh={thresh_val:.0f})"),
        (binary,    f"3. Top {top_percent}% threshold"),
        (clean,     "4. Noise removed"),
        (skeleton,  "5. Skeleton"),
        (cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), "6. Overlay"),
    ]
    for ax, (im, title) in zip(axes.flat, panels):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    plt.savefig("guidewire_pipeline_v3.png", dpi=150, bbox_inches="tight")
    plt.show()

    return wire_mask, skeleton


if __name__ == "__main__":
    import sys
    path         = sys.argv[1] if len(sys.argv) > 1 else "image.png"
    top_percent  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    extract_guidewire(path, top_percent=0.5)