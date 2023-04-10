from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# import struct
CIFAR100_TRAIN_MEAN = (1,1,1)
CIFAR100_TRAIN_STD = (1,1,1)

# Load image
image = Image.open("apple.jpg")

# Define the preprocessing operations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
])

# Apply preprocessing operations
preprocessed_image = transform(image)

# Convert the preprocessed image to a NumPy array
preprocessed_image_np = preprocessed_image.numpy()

# Convert the preprocessed image data to int8
preprocessed_image_int8 = np.floor(preprocessed_image_np * 64 + 0.5).astype(np.int8)
preprocessed_image_int8.tofile("apple_after_resize.bin")

