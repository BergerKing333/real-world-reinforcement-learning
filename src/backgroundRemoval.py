import cv2
import numpy as np

ref_img = cv2.imread('src/background.png')
h, w = ref_img.shape[:2] # Get dimensions

_, mask = cv2.threshold(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# --- NEW: Remove corner data ---
# Define the size of the area you want to clear (e.g., 100x100 pixels)
size = 750

# Top Right Corner: Rows 0 to size, Columns from (width - size) to end
mask[0:size, 0:size] = 255

# Bottom Left Corner: Rows from (height - size) to end, Columns 0 to size
mask[h-size:h, 0:size] = 255


mask[0:h, w-200:w] = 255

# -------------------------------

cv2.imwrite('src/mask.png', mask)