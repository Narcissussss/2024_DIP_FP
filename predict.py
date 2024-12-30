from ultralytics import YOLO
import cv2
import numpy as np
import os

# Paths
model_path = '/home/icssl_pub/Project/Narcissus/DIP_FP/YOLO_v11/runs/segment/train/weights/last.pt'
testing_set_path = '/home/icssl_pub/Project/Narcissus/DIP_FP/YOLO_v11/testing_set/'
output_dir = '/home/icssl_pub/Project/Narcissus/DIP_FP/YOLO_v11/result/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the YOLO model
model = YOLO(model_path)

# Iterate over all .jpg files in the testing set directory
for file_name in os.listdir(testing_set_path):
    if file_name.endswith('.jpg'):  # Process only .jpg files
        image_path = os.path.join(testing_set_path, file_name)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Could not read image: {image_path}")
            continue

        H, W, _ = img.shape

        # Perform prediction
        results = model(img)

        if results is None or len(results) == 0:
            print(f"No results found for image: {file_name}")
            # Output a white image
            white_image = np.full((H, W), 255, dtype=np.uint8)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png")
            cv2.imwrite(output_path, white_image)
            print(f"White mask saved to {output_path}")
            continue

        result = results[0]  # Assuming one result per image
        if result.masks is None or len(result.masks.data) == 0:
            print(f"No mask detected for image: {file_name}")
            # Output a white image
            white_image = np.full((H, W), 255, dtype=np.uint8)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png")
            cv2.imwrite(output_path, white_image)
            print(f"White mask saved to {output_path}")
            continue

        # Process and save the first mask only
        mask = result.masks.data[0].cpu().numpy() * 255  # Convert to 8-bit grayscale
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)  # Resize to match input dimensions

        # Save the mask
        output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png")
        cv2.imwrite(output_path, mask.astype(np.uint8))
        print(f"Mask saved to {output_path}")
