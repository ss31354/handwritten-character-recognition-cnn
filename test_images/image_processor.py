import cv2
import numpy as np
import os

def process_digit_image(input_path, output_path):
    # Load image and validate
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' does not exist")
    
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Failed to load image - check if file is corrupted")

    # Convert to grayscale and binarize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours and select largest one (the digit)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No digit found in the image")
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Extract the digit and make it bold
    digit = gray[y:y+h, x:x+w]
    _, digit_binary = cv2.threshold(digit, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    bold_digit = cv2.dilate(digit_binary, kernel, iterations=2)

    # Resize digit to fit canvas while maintaining aspect ratio
    canvas_width, canvas_height = 1200, 900
    scale = min(canvas_width/w, canvas_height/h) * 0.9  # 90% of available space
    new_w, new_h = int(w*scale), int(h*scale)
    resized_digit = cv2.resize(255 - bold_digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create white canvas and center the digit
    canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255
    center_x = (canvas_width - new_w) // 2
    center_y = (canvas_height - new_h) // 2
    canvas[center_y:center_y+new_h, center_x:center_x+new_w] = resized_digit

    # Save output
    cv2.imwrite(output_path, canvas)
    print(f"Successfully processed and saved to {output_path}")

    # Show result
    cv2.imshow("Processed Digit", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
input_image = "test_images/3.png"  # CHANGE THIS
output_image = "test_images/3_output.png"     # Output filename

process_digit_image(input_image, output_image)