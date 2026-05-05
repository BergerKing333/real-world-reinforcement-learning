import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, skeletonize

def find_guidewire_tip(skeleton_img, spline_point_count=5):
    skel_bin = (skeleton_img > 0).astype(np.uint8)

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = cv2.filter2D(skel_bin, -1, kernel)

    endpoints_mask = (skel_bin == 1) & (neighbor_count == 1)

    y_coords, x_coords = np.where(endpoints_mask)
    endpoints = list(zip(x_coords, y_coords)) 

    if len(endpoints) == 0:
        return None, []

    # Assuming tip is the right-most point
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
            break

    # Safely slice to avoid returning empty lists if the wire is too short
    sampled_spline = spline_points[10::10]
    if not sampled_spline and len(spline_points) > 1:
        sampled_spline = [spline_points[-1]] # Fallback to the last found point

    return tip_position, sampled_spline

def detect_dots(gray: np.ndarray, min_r: int = 15, max_r: int = 45) -> np.ndarray:
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
    return dot_mask

env_mask_raw = cv2.imread('src/environment_mask.png', cv2.IMREAD_GRAYSCALE)
if env_mask_raw is not None:
    environment_mask = cv2.bitwise_not(env_mask_raw)
    trim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    environment_mask = cv2.erode(environment_mask, trim_kernel, iterations=1)
    smooth_mask = cv2.GaussianBlur(environment_mask, (15, 15), 0)
    _, environment_mask = cv2.threshold(smooth_mask, 127, 255, cv2.THRESH_BINARY)
else:
    print("Warning: 'src/environment_mask.png' not found. Masking will be bypassed.")
    environment_mask = None

def segment_guidewire(raw_image):
    img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

    masked_image = img.copy()
    masked_image[environment_mask == 0] = 255

    masked_image = cv2.threshold(masked_image, 100, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('src/masked_image.png', masked_image)

    kernel_size = 25
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(masked_image, cv2.MORPH_BLACKHAT, kernel)

    # Ensure mask matches image size to prevent bitwise_and crashes
    if environment_mask is not None:
        if environment_mask.shape != blackhat.shape:
            local_mask = cv2.resize(environment_mask, (blackhat.shape[1], blackhat.shape[0]))
        else:
            local_mask = environment_mask
        masked_blackhat = cv2.bitwise_and(blackhat, blackhat, mask=local_mask)
    else:
        masked_blackhat = blackhat

    frangi_filtered = frangi(masked_blackhat, sigmas=range(4, 6, 2), black_ridges=False)
    frangi_norm = cv2.normalize(frangi_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    _, binary = cv2.threshold(frangi_norm, 100, 255, cv2.THRESH_BINARY)

    binary_bool = binary > 0
    
    cleaned = remove_small_objects(binary_bool, min_size=400)
    skeleton = skeletonize(cleaned)

    # cleaned_out = img.copy()
    # cleaned_out[skeleton] = 255
    # cv2.imwrite('src/cleaned.png', cleaned_out)

    if len(skeleton) < 100:
        return None

    return skeleton

def generate_goal_points(tip_position, spline_points, obstacle_mask, num_goal_points=1):
    if not spline_points:
        return [] # Safe return if no spline points exist

    approximate_direction = np.array(spline_points[0]) - np.array(tip_position)
    norm = np.linalg.norm(approximate_direction)
    if norm == 0:
        return []
        
    approximate_direction = approximate_direction / norm

    goal_points = []
    for _ in range(1000):
        distance = np.random.uniform(50, 64)
        angle = np.random.uniform(0, 2 * np.pi)

        tmp_goal_point = np.array(tip_position) + [distance * np.cos(angle), distance * np.sin(angle)]
        
        px, py = int(tmp_goal_point[0]), int(tmp_goal_point[1])
        if (0 <= px < obstacle_mask.shape[1] and
            0 <= py < obstacle_mask.shape[0] and
            obstacle_mask[py, px] == 0):
            goal_points.append(tmp_goal_point)
            if len(goal_points) >= num_goal_points:
                break
    return goal_points


if __name__ == "__main__":
    IMAGE_PATH = "tmp/catheter_images/970862213914510.bin"
    IMAGE_PATH = "tmp/catheter_images/1000902266408667.bin"
    IMAGE_PATH = "tmp/catheter_images/1042546816429736.bin"
    h = 2048
    w = 2448

    # 1. Safely load the binary file
    try:
        raw = np.fromfile(IMAGE_PATH, dtype=np.uint8)
    except FileNotFoundError:
        print(f"Error: Could not find file {IMAGE_PATH}")
        exit()

    expected = h * w * 3
    if raw.size != expected:
        print(f"Error: File size ({raw.size}) does not match expected size ({expected}).")
        exit()

    # 2. Reshape and ensure contiguous arrays
    bgr = raw.reshape((h, w, 3))
    rgb = bgr[:, :, ::-1].copy()  # BGR → RGB
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 3. Process
    final_mask = segment_guidewire(rgb)
    end_point, spline_points = find_guidewire_tip(skeleton_img=final_mask)
    obstacle_mask = detect_dots(gray) # Use the array in memory, not cv2.imread

    # 4. Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.axis('off')
    
    # Use 'rgb' array directly for the background
    ax.imshow(rgb)
    ax.imshow(final_mask, cmap='jet', alpha=1.0 * (final_mask > 0))

    if end_point is not None:
        ax.plot(end_point[0], end_point[1], 'ro', markersize=5)
    
    if spline_points:
        for pt in spline_points:
            ax.plot(pt[0], pt[1], 'go', markersize=3)
            
    ax.imshow(obstacle_mask, cmap='Reds', alpha=0.5 * (obstacle_mask > 0))

    goal_points = generate_goal_points(end_point, spline_points, obstacle_mask, 10)
    for gp in goal_points:
        ax.plot(gp[0], gp[1], 'bx', markersize=5)

    plt.title("Segmented Guidewire with Detected Tip")
    plt.show()