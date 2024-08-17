import cv2
import numpy as np

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

# Initialize a filter index
filter_idx = 0
filters = [apply_sketch_filter, apply_cartoon_filter, apply_emboss_filter]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the current filter based on filter_idx
    filtered_frame = filters[filter_idx](frame)

    # Display the frame with the applied filter
    cv2.imshow('Artistic Filters', filtered_frame)

    # Change filter on 'f' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('f'):
        filter_idx = (filter_idx + 1) % len(filters)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()