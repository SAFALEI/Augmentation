import os
from collections import defaultdict


def count_class_instances(label_folder):
    class_counts = defaultdict(int)

    for label_file in os.listdir(label_folder):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_folder, label_file), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    class_counts[class_id] += 1

    return class_counts


def print_class_counts(class_counts):
    print("Class ID\tCount")
    print("----------------")
    for class_id, count in sorted(class_counts.items()):
        print(f"{class_id}\t{count}")


def main():
    # Path configuration
    augmented_dataset_path = expand_user_path('/home/adas/safaei/Augmentation/')

    # Process train, val, and test folders
    for folder in ['train', 'val', 'test']:
        label_folder = os.path.join(augmented_dataset_path, folder, 'labels')

        if not os.path.isdir(label_folder):
            print(f"Label folder not found: {label_folder}")
            continue

        print(f"Processing {folder} dataset:")
        class_counts = count_class_instances(label_folder)
        print_class_counts(class_counts)
        print()


def expand_user_path(path):
    return os.path.expanduser(path)


if __name__ == "__main__":
    main()
