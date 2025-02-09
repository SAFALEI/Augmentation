import albumentations as A
import cv2

# Declare an augmentation pipeline
transform = A.Compose([
    # A.RandomCrop(width=256, height=256),
    # A.HorizontalFlip(p=0.5),
    A.Spatter(p=1),
])

# Read an image with OpenCV and convert it to the RGB colorspace
image = cv2.imread("262.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Augment an image
transformed = transform(image=image)
transformed_image = transformed["image"]
transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
transformed_image=cv2.resize(transformed_image,(640,400))
cv2.imwrite("transformed_image.png",transformed_image)
cv2.imshow("image", transformed_image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image=cv2.resize(image,(640,400))
cv2.imwrite("original.png",image)
cv2.imshow("original", image)
cv2.waitKey(0)

print("edited in origin")
