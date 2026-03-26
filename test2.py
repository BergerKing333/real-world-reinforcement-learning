import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, skeletonize
import time

def find_guidewire_tip(skeleton_img):
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

    return tip_position # Returns (x, y)

def segment_guidewire(image_path):
    # 1. Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
    
    cleaned = remove_small_objects(binary_bool, min_size=150)
    
    # Thin the wire down to a 1-pixel wide line
    skeleton = skeletonize(cleaned)

    end_point = find_guidewire_tip(skeleton_img=skeleton)

    print(f"Processing time: {end - start:.2f} seconds")

    # --- Plotting the Results ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax = axes.ravel()

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Grayscale')

    ax[1].imshow(blackhat, cmap='gray')
    ax[1].set_title('Black-Hat Transform')

    ax[2].imshow(frangi_filtered, cmap='hot')
    ax[2].set_title('Frangi Filter Output')

    ax[3].imshow(binary, cmap='gray')
    ax[3].set_title('Otsu Threshold')

    ax[4].imshow(cleaned, cmap='gray')
    ax[4].set_title('Noise Removal')

    ax[5].imshow(skeleton, cmap='gray')
    ax[5].set_title('Final Skeleton')

    if end_point is not None:
        ax[5].plot(end_point[0], end_point[1], 'ro', markersize=5)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

    return skeleton

# Run the function
# Replace 'guidewire.jpg' with the actual filename of your image
final_mask = segment_guidewire('image.png')