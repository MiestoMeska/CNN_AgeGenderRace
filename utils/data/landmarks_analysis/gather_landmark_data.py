import face_alignment
import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

def gather_landmark_data(df, image_folder, num_images=None):
    """
    Process images to detect facial landmarks and bounding boxes using FAN,
    and return a DataFrame with filename, landmark data, and face-to-image ratios.
    
    Args:
        df (pd.DataFrame): DataFrame containing the image filenames (in 'filename' column).
        image_folder (str): Path to the folder containing the images.
        num_images (int): Number of images to process. If None, process all images.
    
    Returns:
        pd.DataFrame: DataFrame with 'filename', bounding box, landmark data, and face-to-image ratio.
    """
    if num_images is not None:
        df = df.sample(n=num_images)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_path = os.path.join(image_folder, row['filename'])
        
        try:
            img = np.array(Image.open(image_path))
            img_height, img_width = img.shape[:2]

            preds = fa.get_landmarks(img)
            
            if preds is not None:
                landmarks = preds[0]
                
                left_eye = landmarks[36]
                right_eye = landmarks[45]
                nose = landmarks[30]
                left_mouth = landmarks[48]
                right_mouth = landmarks[54]
                
                left = min([point[0] for point in landmarks])
                top = min([point[1] for point in landmarks])
                right = max([point[0] for point in landmarks])
                bottom = max([point[1] for point in landmarks])
                bounding_box = [left, top, right, bottom]
                
                distance_left = left / img_width
                distance_top = top / img_height
                distance_right = (img_width - right) / img_width
                distance_bottom = (img_height - bottom) / img_height
                
                eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

                normalized_left_eye = (left_eye[0] / img_width, left_eye[1] / img_height)
                normalized_right_eye = (right_eye[0] / img_width, right_eye[1] / img_height)
                
                normalized_eye_midpoint = (
                    (normalized_left_eye[0] + normalized_right_eye[0]) / 2,
                    (normalized_left_eye[1] + normalized_right_eye[1]) / 2
                )

                eye_midpoint = (
                    (left_eye[0] + right_eye[0]) / 2,
                    (left_eye[1] + right_eye[1]) / 2
                )

                normalized_nose = (nose[0] / img_width, nose[1] / img_height)
                normalized_left_mouth = (left_mouth[0] / img_width, left_mouth[1] / img_height)
                normalized_right_mouth = (right_mouth[0] / img_width, right_mouth[1] / img_height)
                
                result = {
                    'filename': row['filename'],
                    'bounding_box': bounding_box,
                    'left_eye': left_eye,
                    'right_eye': right_eye,
                    'nose': nose,
                    'left_mouth': left_mouth,
                    'right_mouth': right_mouth,
                    'distance_left': distance_left,
                    'distance_top': distance_top,
                    'distance_right': distance_right,
                    'distance_bottom': distance_bottom,
                    'eye_distance': eye_distance,
                    'eye_midpoint': eye_midpoint,
                    'normalized_left_eye': normalized_left_eye,
                    'normalized_right_eye': normalized_right_eye,
                    'normalized_eye_midpoint': normalized_eye_midpoint,
                    'normalized_nose': normalized_nose,
                    'normalized_left_mouth': normalized_left_mouth,
                    'normalized_right_mouth': normalized_right_mouth
                }
            else:
                result = {
                    'filename': row['filename'],
                    'bounding_box': None,
                    'left_eye': None,
                    'right_eye': None,
                    'nose': None,
                    'left_mouth': None,
                    'right_mouth': None,
                    'distance_left': None,
                    'distance_top': None,
                    'distance_right': None,
                    'distance_bottom': None,
                    'eye_distance': None,
                    'eye_midpoint': None,
                    'normalized_left_eye': None,
                    'normalized_right_eye': None,
                    'normalized_eye_midpoint': None,
                    'normalized_nose': None,
                    'normalized_left_mouth': None,
                    'normalized_right_mouth': None,
                    'eye_midpoint': None
                }
                
        except Exception as e:
            result = {
                'filename': row['filename'],
                'bounding_box': None,
                'left_eye': None,
                'right_eye': None,
                'nose': None,
                'left_mouth': None,
                'right_mouth': None,
                'distance_left': None,
                'distance_top': None,
                'distance_right': None,
                'distance_bottom': None,
                'eye_distance': None,
                'normalized_left_eye': None,
                'normalized_right_eye': None,
                'normalized_nose': None,
                'normalized_left_mouth': None,
                'normalized_right_mouth': None,
                'eye_midpoint': None
            }
            print(f"Error processing {image_path}: {e}")

        results.append(result)
    
    landmarks_df = pd.DataFrame(results)
    
    return landmarks_df
