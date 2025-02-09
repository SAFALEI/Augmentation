import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
from math import ceil


# Define your augmentation pipeline
def get_transform():
    return A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=10, p=0.3),
        A.Resize(height=1200, width=1920, p=1.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=50, p=0.5),
        A.GaussNoise(var_limit=(20.0, 50.0), p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# Function to expand user directory (~)
def expand_user_path(path):
    return os.path.expanduser(path)


# Path configuration
dataset_path = expand_user_path('/home/adas/safaei/Augmentation/Test_dataset')
output_path = expand_user_path('/home/adas/safaei/Augmentation/augmented_dataset')
os.makedirs(output_path, exist_ok=True)


# Function to get class counts
def get_class_counts(label_folder):
    class_counts = defaultdict(int)
    for label_file in os.listdir(label_folder):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_folder, label_file), 'r') as file:
                for line in file:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    return class_counts


# Function to get the maximum class count
def get_max_class_count(class_counts):
    """Return the maximum count and the class ID(s) with that count."""
    max_count = max(class_counts.values())
    max_classes = [class_id for class_id, count in class_counts.items() if count == max_count]
    return max_count, max_classes


# Function to convert tensor to numpy array
def tensor_to_numpy(tensor):
    return tensor.permute(1, 2, 0).cpu().numpy()
print("edited in local 3")


print("edit in origin 4")
# Function to augment dataset for a given class
def augment_class(input_folder, output_folder, transform, target_class_count):
    os.makedirs(output_folder, exist_ok=True)

    image_folder = os.path.join(input_folder, 'images')
    label_folder = os.path.join(input_folder, 'labels')

    output_image_folder = os.path.join(output_folder, 'images')
    output_label_folder = os.path.join(output_folder, 'labels')

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    # Track the number of augmented images
    augmented_counts = defaultdict(int)
    processed_images = set()  # Keep track of processed images to ensure only one augmentation per image

    for img_name in os.listdir(image_folder):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            img_path = os.path.join(image_folder, img_name)
            label_path = os.path.join(label_folder, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

            if not os.path.isfile(label_path):
                print(f"Label file not found: {label_path}")
                continue

            image = cv2.imread(img_path)
            h, w, _ = image.shape

            bboxes = []
            class_labels = []
            with open(label_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    class_label = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_label)

            # Get the class IDs with the maximum count
            max_class_count, max_classes = get_max_class_count(class_counts)

            # Calculate how many times this image needs to be augmented
            for class_id in set(class_labels):
                if class_counts[class_id] < target_class_count:
                    # Skip augmentation for the class with the maximum instances
                    # if class_counts[class_id] == target_class_count:
                    #     continue
                    if class_id in max_classes:
                        continue
                    # Determine how many times to augment the image based on the ratio
                    current_class_count = class_counts[class_id]
                    ratio = ceil(target_class_count / current_class_count)

                    # Counter for augmented images
                    augmentation_counter = 1

                    while augmented_counts[class_id] < target_class_count:
                        #if img_name in processed_images:
                        #    break  # Skip processing if this image has been processed already

                        # Apply augmentations
                        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                        augmented_image = transformed['image']
                        augmented_bboxes = transformed['bboxes']
                        augmented_class_labels = transformed['class_labels']

                        # Convert tensor to numpy array if needed
                        if isinstance(augmented_image, np.ndarray):
                            save_image = augmented_image
                        else:
                            save_image = tensor_to_numpy(augmented_image)

                        # Ensure the image is in the correct format and size
                        save_image = (save_image * 255).astype(
                            np.uint8) if save_image.dtype == np.float32 else save_image
                        if save_image.shape[:2] != (1200, 1920):
                            save_image = cv2.resize(save_image, (1920, 1200))  # Resize to ensure final dimensions

                        # Create a unique suffix for each augmented image
                        img_basename, img_ext = os.path.splitext(img_name)
                        out_img_name = f"{img_basename}_{augmentation_counter}{img_ext}"

                        # Save augmented image
                        out_img_path = os.path.join(output_image_folder, out_img_name)
                        cv2.imwrite(out_img_path, save_image)

                        # Save augmented labels
                        out_label_name = f"{img_basename}_{augmentation_counter}.txt"
                        out_label_path = os.path.join(output_label_folder, out_label_name)
                        with open(out_label_path, 'w') as file:
                            for bbox, label in zip(augmented_bboxes, augmented_class_labels):
                                file.write(f"{label} {' '.join(map(str, bbox))}\n")

                        # Increment augmented count for this class
                        augmented_counts[class_id] += 1
                        processed_images.add(img_name)  # Mark this image as processed
                        augmentation_counter += 1

                        # Break if the target count for the current class is reached
                        if augmented_counts[class_id] >= target_class_count:
                            break


# Process train, val, and test folders
for folder in ['train', 'val', 'test']:
    input_folder = os.path.join(dataset_path, folder)
    output_folder = os.path.join(output_path, folder)

    if not os.path.isdir(input_folder):
        print(f"Input folder not found: {input_folder}")
        continue

    class_counts = get_class_counts(os.path.join(input_folder, 'labels'))
    max_class_count, max_classes = get_max_class_count(class_counts)

    transform = get_transform()
    augment_class(input_folder, output_folder, transform, max_class_count)
