import cv2 
import numpy as np

# Load image
image = cv2.imread('sonar_results/position_-25.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(image, (9, 9), 0)

# Detect circles representing the pipe
circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT_ALT , 1.5 , 100 ,param1=30 ,param2=0.1, minRadius=40 ,maxRadius=100)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(blurred, (x, y), r, (200, 50, 100), 4)
        # Save the image
    cv2.imwrite('sonar_results/position_-25_processed_with_circles.png', blurred)
else: 
    print("No circles detected")
    

