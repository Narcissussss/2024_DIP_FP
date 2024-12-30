import cv2
import numpy as np
import os

# Define the class label for 'water'
class_label = 0  # Adjust this based on your class indexing

# Paths to input mask images and output directory for YOLO labels
input_dirs = [
    '/home/icssl_pub/Project/Narcissus/DIP_FP/YOLO_v11/raw_label/train/',
    '/home/icssl_pub/Project/Narcissus/DIP_FP/YOLO_v11/raw_label/val/'
]

output_dirs = [
    '/home/icssl_pub/Project/Narcissus/DIP_FP/YOLO_v11/train/labels',
    '/home/icssl_pub/Project/Narcissus/DIP_FP/YOLO_v11/val/labels'
]

# Ensure output directories exist
for output_label_dir in output_dirs:
    os.makedirs(output_label_dir, exist_ok=True)

# Iterate over input directories (train and val)
for input_mask_dir, output_label_dir in zip(input_dirs, output_dirs):
    # Iterate over each mask image
    for mask_filename in os.listdir(input_mask_dir):
        if mask_filename.endswith('.png'):  # Adjust extension if necessary
            # Load the binary mask image
            mask_path = os.path.join(input_mask_dir, mask_filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Prepare the YOLO label content
            yolo_label = []

            for cnt in contours:
                # Calculate epsilon for contour approximation
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Normalize coordinates
                height, width = mask.shape
                normalized_points = []
                for point in approx:
                    x = point[0][0] / width
                    y = point[0][1] / height
                    normalized_points.append(f'{x:.6f} {y:.6f}')

                # Create a line for each contour
                if normalized_points:
                    yolo_label.append(f"{class_label} " + " ".join(normalized_points))

            # Save the YOLO label to a text file
            label_filename = os.path.splitext(mask_filename)[0] + '.txt'
            label_path = os.path.join(output_label_dir, label_filename)
            with open(label_path, 'w') as file:
                file.write("\n".join(yolo_label))
