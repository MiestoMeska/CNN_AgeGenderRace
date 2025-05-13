import face_alignment
import cv2
import os
import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

def detect_landmarks_fan(image_path):
    """
    Detect facial landmarks using FAN and draw them on the image.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        output_img (numpy array): Image with landmarks drawn.
    """
    img = cv2.imread(image_path)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    preds = fa.get_landmarks(img_rgb)
    
    if preds is not None:
        for landmarks in preds:
            for (x, y) in landmarks:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
    
    return img

def process_images_with_landmarks(input_folder, output_folder, num_images=100, num_samples=64):
    """
    Detect landmarks using FAN and save images with landmarks drawn.
    Displays a grid of randomly selected images with landmarks drawn after processing.

    Args:
        input_folder (str): Path to the folder containing the input images.
        output_folder (str): Path to the folder where output images will be saved.
        num_images (int): Number of images to process and save. If None, process all images.
        num_samples (int): Number of images to display on a grid (8x8).
    """
    image_files = os.listdir(input_folder)
    
    if num_images is not None:
        image_files = image_files[:num_images]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in tqdm(image_files, desc="Processing images"):
        input_image_path = os.path.join(input_folder, image_file)
        
        output_img = detect_landmarks_fan(input_image_path)
        
        output_image_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_image_path, output_img)

    sampled_images = random.sample(image_files, num_samples)
    
    fig, axes = plt.subplots(8, 8, figsize=(15, 15))
    axes = axes.ravel()

    for i, image_file in enumerate(sampled_images):
        img = cv2.imread(os.path.join(output_folder, image_file))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()