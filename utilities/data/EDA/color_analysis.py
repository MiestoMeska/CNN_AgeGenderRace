import os
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from tqdm import tqdm

def is_grayscale(image):
    arr = np.array(image)
    if len(arr.shape) < 3 or arr.shape[2] == 1:
        return True
    return np.allclose(arr[..., 0], arr[..., 1]) and np.allclose(arr[..., 1], arr[..., 2])

def average_rgb(image):
    arr = np.array(image).astype(np.float32)
    return arr.mean(axis=(0, 1))

def average_hsv(image):
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    return hsv.mean(axis=(0, 1))  # (Hue, Saturation, Value)

def analyze_color_patterns(df_labels, image_folder):
    color_stats = []

    for i, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc="Analyzing color"):
        img_path = os.path.join(image_folder, row['filename'])
        try:
            image = Image.open(img_path).convert('RGB')
            avg_r, avg_g, avg_b = average_rgb(image)
            avg_h, avg_s, avg_v = average_hsv(image)
            grayscale = is_grayscale(image)
            
            color_stats.append({
                'filename': row['filename'],
                'age': row['age'],
                'gender': row['gender'],
                'race': row['race'],
                'avg_r': avg_r,
                'avg_g': avg_g,
                'avg_b': avg_b,
                'avg_hue': avg_h,
                'avg_saturation': avg_s,
                'avg_brightness': avg_v,
                'is_grayscale': grayscale
            })
        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")

    return pd.DataFrame(color_stats)
