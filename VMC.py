import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
hand_detector = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

def within_pixel_distance(x1, y1, x2, y2, distance=50):
    return abs(x1 - x2) < distance and abs(y1 - y2) < distance

# Variables to track the position and debounce
index_x, index_y = 0, 0
thumb_x, thumb_y = 0, 0
middle_x, middle_y = 0, 0
ring_x, ring_y = 0, 0
pinky_x, pinky_y = 0, 0
click_debounce_time = 0.2  # 200 ms debounce time for clicks
last_click_time = time.time()

# Smoothing parameters
smooth_factor = 0.8
prev_index_x, prev_index_y = 0, 0

while True:
    # Read and flip the frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS,
                                         drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                         drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
            landmarks = hand.landmark
            fingers_up = []  # Track which fingers are up (1 for up, 0 for down)
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                if id == 8:  # Index finger tip
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y
                    # Smooth movement
                    index_x = smooth_factor * prev_index_x + (1 - smooth_factor) * index_x
                    index_y = smooth_factor * prev_index_y + (1 - smooth_factor) * index_y
                    prev_index_x, prev_index_y = index_x, index_y
                
                if id == 4:  # Thumb tip
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y
                
                if id == 12:  # Middle finger tip
                    middle_x = screen_width / frame_width * x
                    middle_y = screen_height / frame_height * y
                
                if id == 16:  # Ring finger tip
                    ring_x = screen_width / frame_width * x
                    ring_y = screen_height / frame_height * y
                
                if id == 20:  # Pinky finger tip
                    pinky_x = screen_width / frame_width * x
                    pinky_y = screen_height / frame_height * y
                
                # Check the positions of the tips relative to the PIP joints to determine if a finger is up
                if id in [8, 12, 16, 20]:  # Tips of index, middle, ring, and pinky fingers
                    pip_y = landmarks[id - 2].y * frame_height
                    fingers_up.append(1 if y < pip_y else 0)
    
            current_time = time.time()
            
            # Left Click: Index and Thumb fingers close
            if within_pixel_distance(index_x, index_y, thumb_x, thumb_y, 50) and (current_time - last_click_time > click_debounce_time):
                pyautogui.click()
                last_click_time = current_time
            
            # Right Click: Thumb and Middle fingers close
            elif within_pixel_distance(thumb_x, thumb_y, middle_x, middle_y, 50) and (current_time - last_click_time > click_debounce_time):
                pyautogui.rightClick()
                last_click_time = current_time
            
            # Move the cursor: Only if Index and Middle fingers are close and others are down
            if within_pixel_distance(index_x, index_y, middle_x, middle_y, 50) and fingers_up == [1, 1, 0, 0]:
                pyautogui.moveTo(index_x, index_y)
    
    # Display the frame
    cv2.imshow('Virtual Mouse', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()