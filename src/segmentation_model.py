import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, skeletonize
import time

def find_guidewire_tip(skeleton_img, spline_point_count=5):
    skel_bin = (skeleton_img > 0).astype(np.uint8)

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = cv2.filter2D(skel_bin, -1, kernel)

    endpoints_mask = (skel_bin == 1) & (neighbor_count == 1)

    y_coords, x_coords = np.where(endpoints_mask)
    endpoints = list(zip(x_coords, y_coords)) # Group into (x, y) tuples

    if len(endpoints) == 0:
        return None

    tip_position = max(endpoints, key=lambda pt: pt[0]) 

    spline_points = [tip_position]
    current = tip_position
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for i in range(0, (spline_point_count - 1) * 10):
        found_next = False
        for dx, dy in neighbors:
            next_x, next_y = current[0] + dx, current[1] + dy
            if (0 <= next_x < skel_bin.shape[1] and
                0 <= next_y < skel_bin.shape[0] and
                skel_bin[next_y, next_x] == 1 and
                (next_x, next_y) not in spline_points):
                current = (next_x, next_y)
                spline_points.append(current)
                found_next = True
                break
        if not found_next:
            # print(f"  [spline] stopped at point {current}")
            break

    return tip_position, spline_points[10::10] # Returns (x, y) and the list of spline points

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
    # print(f"  [dots] detected {0 if circles is None else len(circles[0])} circles")
    return dot_mask

def segment_guidewire(raw_image):
    # 1. Load the image in grayscale
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    kernel_size = 25  # Should be larger than the wire's width
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    start = time.time()

    frangi_filtered = frangi(blackhat, sigmas=range(2, 4, 2), black_ridges=False)
    end = time.time()

    frangi_norm = cv2.normalize(frangi_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary = cv2.threshold(frangi_norm, 40, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_bool = binary > 0
    
    cleaned = remove_small_objects(binary_bool, max_size=150)
    
    # Thin the wire down to a 1-pixel wide line
    skeleton = skeletonize(cleaned)

    return skeleton

def generate_goal_points(tip_position, spline_points, obstacle_mask, num_goal_points=1):
    approximate_direction = np.array(spline_points[0]) - np.array(tip_position)
    approximate_direction = approximate_direction / np.linalg.norm(approximate_direction)

    # print(obstacle_mask)
    goal_points = []
    for i in range(1000):
        distance = np.random.uniform(30, 50)
        angle = np.random.uniform(0, 2 * np.pi)

        tmp_goal_point = np.array(tip_position) + [distance * np.cos(angle), distance * np.sin(angle)]
        # print(tmp_goal_point)
        if (0 <= tmp_goal_point[0] < obstacle_mask.shape[1] and
            0 <= tmp_goal_point[1] < obstacle_mask.shape[0] and
            obstacle_mask[int(tmp_goal_point[1]), int(tmp_goal_point[0])] == 0):
            goal_points.append(tmp_goal_point)
            if len(goal_points) >= num_goal_points:
                break
    return goal_points


# Run the function
# Replace 'guidewire.jpg' with the actual filename of your image
if __name__ == "__main__":
    IMAGE_PATH = 'tmp/catheter_images/1210728504099.bin'
    raw = np.fromfile(IMAGE_PATH, dtype=np.uint8)
    # expected = 2048 * 2448 * 3
    # if raw.size != expected:
    #     return np.zeros((h, w, 3), dtype=np.uint8)
    bgr = raw.reshape((2048, 2448, 3))
    rgb = bgr[:, :, ::-1]  # BGR → RGB
    final_mask = segment_guidewire(rgb)

    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.axis('off')
    

    end_point, spline_points = find_guidewire_tip(skeleton_img=final_mask)
    ax.imshow(cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR))
    ax.imshow(final_mask, cmap='jet', alpha=1.0 * (final_mask > 0))

    if end_point is not None:
        ax.plot(end_point[0], end_point[1], 'ro', markersize=5)
    if spline_points:
        for pt in spline_points:
            ax.plot(pt[0], pt[1], 'go', markersize=3)
    obstacle_mask = detect_dots(cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE))
    ax.imshow(obstacle_mask, cmap='Reds', alpha=0.5 * (obstacle_mask > 0))

    goal_points = generate_goal_points(end_point, spline_points, obstacle_mask, 10)
    for gp in goal_points:
        ax.plot(gp[0], gp[1], 'bx', markersize=5)

    plt.title("Segmented Guidewire with Detected Tip")
    plt.show()
