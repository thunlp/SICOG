import requests
from PIL import Image
from io import BytesIO
import random
import torchvision.transforms as T
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
import imageio

import cv2


def apply_gaussian_blur(image):

    image_np = np.array(image)
    augmentation = iaa.GaussianBlur(sigma=(3.0, 5.0))
    blurred_image = augmentation(images=np.array([image_np]))

    return Image.fromarray(np.uint8(blurred_image[0]))


def apply_color_jitter(image):

    jitter = T.ColorJitter(
        brightness=0.8,
        contrast=0.8,
        saturation=0.8,
        hue=0.2
    )
    return jitter(image)



def apply_mirror_rotation(image):
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    rotation_angle = random.choice([90, 270])
    return image.rotate(rotation_angle, expand=True)


# step4 Missing Information
def apply_mask(image, mask_ratio=0.30, tolerance=0.05):
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    times = 0
    while True:
        times += 1

        angle = random.uniform(0, 360)
        theta = np.radians(angle)

        x0 = random.randint(0, width)
        y0 = random.randint(0, height)
        
        mask = np.zeros((height, width), dtype=bool)
        
        for x in range(width):
            for y in range(height):
                dist = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
                if dist > 0:
                    mask[y, x] = True
        
        ratio1 = np.sum(mask) / (height * width)
        ratio2 = np.sum(~mask) / (height * width)
        
        if abs(ratio1 - mask_ratio) <= tolerance or abs(ratio2 - mask_ratio) <= tolerance:
            if abs(ratio1 - mask_ratio) <= abs(ratio2 - mask_ratio):
                final_mask = mask
            else:
                final_mask = ~mask
            break

    masked_region = img_array[final_mask]
    avg_color = np.mean(masked_region, axis=0).astype(np.uint8)
    
    result = img_array.copy()
    result[final_mask] = avg_color
    
    result_image = Image.fromarray(result)
    
    return result_image



