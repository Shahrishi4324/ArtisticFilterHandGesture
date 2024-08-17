import cv2
import numpy as np
import mediapipe as mp

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Define the filters
def apply_sketch_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv_gray = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inv_gray, (21, 21), 0)
    inverted_blur = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def apply_cartoon_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_emboss_filter(frame):
    kernel = np.array([[0,-1,-1], [1,0,-1], [1,1,0]])
    embossed = cv2.filter2D(frame, -1, kernel)
    return embossed

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Add hand gesture recognition to switch filters
def detect_hand_gesture(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Detect a specific gesture to switch filter (e.g., thumbs up)
            landmarks = hand_landmarks.landmark
            if landmarks[4].y < landmarks[3].y:  # Thumb is up
                return True
    return False

# Function to add overlay with instructions
def add_overlay(frame, filter_name):
    cv2.putText(frame, f'Filter: {filter_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, 'Gesture: Thumbs Up to Switch Filters', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# Initialize a filter index
filter_idx = 0
filters = [(apply_sketch_filter, "Sketch"), (apply_cartoon_filter, "Cartoon"), (apply_emboss_filter, "Emboss")]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect hand gestures
    gesture_detected = detect_hand_gesture(frame)

    # Switch filter if gesture is detected
    if gesture_detected:
        filter_idx = (filter_idx + 1) % len(filters)

    # Apply the current filter
    filter_function, filter_name = filters[filter_idx]
    filtered_frame = filter_function(frame)

    # Add overlay with instructions
    overlay_frame = add_overlay(filtered_frame, filter_name)

    # Display the frame with the applied filter and gesture overlay
    cv2.imshow('Artistic Filters with Gesture Control', overlay_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()