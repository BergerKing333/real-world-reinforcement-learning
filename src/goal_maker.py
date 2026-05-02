import cv2

# --- 1. Define Your Labeling Modes ---
MODES = ["Main Branch", "Sub-branch", "End Node", "Error Marker"]
current_mode_idx = 0

# --- 2. Define Button Position & Size (x1, y1, x2, y2) ---
BTN_X1, BTN_Y1 = 10, 10
BTN_X2, BTN_Y2 = 200, 50

def draw_ui(base_image):
    """Draws the fake button and current mode text on top of the image."""
    # We make a copy so we don't permanently bake the button into the image data
    display_img = base_image.copy()
    
    # Draw button background (Light Gray)
    cv2.rectangle(display_img, (BTN_X1, BTN_Y1), (BTN_X2, BTN_Y2), (200, 200, 200), -1)
    # Draw button border (Black)
    cv2.rectangle(display_img, (BTN_X1, BTN_Y1), (BTN_X2, BTN_Y2), (0, 0, 0), 2)
    
    # Write the current mode on the button
    mode_text = f"Mode: {MODES[current_mode_idx]}"
    cv2.putText(display_img, mode_text, (BTN_X1 + 10, BTN_Y1 + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    return display_img

colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

def click_event(event, x, y, flags, params):
    global current_mode_idx, img, display_img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the user clicked INSIDE the button coordinates
        if BTN_X1 <= x <= BTN_X2 and BTN_Y1 <= y <= BTN_Y2:
            # Toggle to the next mode, loop back to 0 if at the end of the list
            current_mode_idx = (current_mode_idx + 1)
            print(f"--- Switched to mode: {MODES[current_mode_idx]} ---")
            
            # Redraw the UI to update the button text
            display_img = draw_ui(img)
            cv2.imshow('Labeling Tool', display_img)
            
        else:
            # User clicked outside the button, log the coordinate
            current_mode = MODES[current_mode_idx]
            print(f"[{current_mode}] X: {x}, Y: {y}")

            all_goal_points[current_mode_idx].append((x, y))
            
            # Draw a dot where they clicked (Green)
            cv2.circle(img, (x, y), radius=6, color=colors[mode_index % len(colors)], thickness=-1)
            
            # Redraw the UI so the new dot shows up
            display_img = draw_ui(img)
            cv2.imshow('Labeling Tool', display_img)

# --- Main Script Execution ---
image_path = 'src/guidewire_result_v4.png' # Update with your image name
img = cv2.imread(image_path)

all_goal_points = [[]]
mode_index = 0

if img is None:
    print("Error: Could not load image.")
else:
    # Set up the initial view
    display_img = draw_ui(img)
    cv2.namedWindow('Labeling Tool', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Labeling Tool', img.shape[1] // 2, img.shape[0] // 2)  # Resize window to fit the image


    cv2.imshow('Labeling Tool', display_img)

    
    # Attach the click listener
    cv2.setMouseCallback('Labeling Tool', click_event)
    
    print("Labeling Tool Started!")
    print("- Click the button in the top left to change modes.")
    print("- Or press 'm' on your keyboard to toggle modes quickly.")
    print("- Press 'q' or 'ESC' to quit.")
    
    # Keep the window open and listen for key presses
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' or 'ESC' to exit
        if key == 27 or key == ord('q'): 
            break
        # Press 'm' to toggle modes via keyboard
        elif key == ord('m'):
            mode_index += 1
            all_goal_points.append([])  # Start a new list for the next mode
            current_mode_idx = (current_mode_idx + 1) % len(MODES)
            print(f"--- Switched to mode: {MODES[current_mode_idx]} ---")
            display_img = draw_ui(img)
            cv2.imshow('Labeling Tool', display_img)

        elif key == ord('z'):
            if all_goal_points[mode_index]:
                removed_point = all_goal_points[mode_index].pop()  # Remove the last point from the current mode
                print(f"Removed last point: {removed_point} from mode: {MODES[current_mode_idx]}")
                
                # Redraw the image without the removed point
                img = cv2.imread(image_path)  # Reload original image
                for idx, points in enumerate(all_goal_points):
                    for pt in points:
                        cv2.circle(img, pt, radius=6, color=colors[idx % len(colors)], thickness=-1)
                display_img = draw_ui(img)
                cv2.imshow('Labeling Tool', display_img)
            else:
                print(f"No points to remove in mode: {MODES[current_mode_idx]}")

    print(all_goal_points)
    # Clean up
    cv2.destroyAllWindows()