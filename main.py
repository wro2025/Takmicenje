import cv2 as cv
import numpy as np

# Load the image
img = cv.imread("images/test.png")

# Convert to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Define color ranges for red, green, and black
lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
lower_green, upper_green = np.array([35, 50, 50]), np.array([85, 255, 255])
lower_black, upper_black = np.array([0, 0, 0]), np.array([180, 255, 50])

# Create masks
mask_red = cv.bitwise_or(cv.inRange(hsv, lower_red1, upper_red1), cv.inRange(hsv, lower_red2, upper_red2))
mask_green = cv.inRange(hsv, lower_green, upper_green)
mask_black = cv.inRange(hsv, lower_black, upper_black)

# Focal length and real object width (set based on calibration)
KNOWN_DISTANCE = 50  # cm (distance at which reference object was measured)
KNOWN_WIDTH = 10  # cm (actual width of the reference object)
FOCAL_LENGTH = 500  # Adjust based on calibration

# Function to estimate distance
def estimate_distance(object_width):
    if object_width == 0:
        return float("inf")  # Avoid division by zero
    return (KNOWN_WIDTH * FOCAL_LENGTH) / object_width

# Function to find contours, centroids, and distance
def get_object_data(mask, color_name, outline_color):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    object_data = []

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)  # Bounding box (x, y, width, height)
        cx, cy = x + w // 2, y + h // 2  # Centroid coordinates
        distance = estimate_distance(w)  # Estimate distance from width

        object_data.append((cx, cy, distance))

        # Draw bounding box & centroid
        cv.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 2)  # White bounding box
        cv.circle(output, (cx, cy), 5, outline_color, -1)  # Draw centroid
        cv.putText(output, f"{color_name}: {int(distance)}cm", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, outline_color, 2)

        print(f"{color_name} Object at: x={cx}, y={cy}, Distance={distance:.2f}cm")

    return object_data

# Process contours for each color
output = img.copy()
red_objects = get_object_data(mask_red, "Red", (255, 0, 0))  # Red → Blue outline
green_objects = get_object_data(mask_green, "Green", (0, 255, 0))  # Green → Green outline
black_objects = get_object_data(mask_black, "Black", (255, 255, 255))  # Black → White outline

# Determine car movement decision based on position
frame_center = img.shape[1] // 2
left, right = 0, 0

for cx, cy, distance in red_objects + green_objects + black_objects:
    if cx < frame_center - 50:
        left += 1
    elif cx > frame_center + 50:
        right += 1

if left > right:
    decision = "TURN RIGHT"
elif right > left:
    decision = "TURN LEFT"
else:
    decision = "GO STRAIGHT"

print(f"Decision: {decision}")

# Display decision on the image
cv.putText(output, decision, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
cv.imshow("Detected Contours", output)

cv.waitKey(0)
cv.destroyAllWindows()
