import face_alignment
import os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda' if torch.cuda.is_available() else 'cpu')

def process_image_fan(image_path):
    """
    Helper function to process an image and detect face landmarks using FAN.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        int: 1 if a face was found, 0 otherwise.
    """
    try:
        img = np.array(Image.open(image_path))
        preds = fa.get_landmarks(img)

        if preds is not None and len(preds) > 0:
            return len(preds)
        else:
            return 0
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0

def capture_faces_fan(folder_path, df):
    """
    Detect and mark faces in images using FAN with GPU support via PyTorch.
    
    Args:
        folder_path (str): The path to the folder containing the image files.
        df (pd.DataFrame): The DataFrame containing the 'filename' column.
    
    Returns:
        pd.DataFrame: Updated DataFrame with a 'face_found' column (1 for face detected, 0 for no face).
    """
    df = df.copy()

    df['face_found'] = 0

    image_paths = [os.path.join(folder_path, file_name) for file_name in df['filename']]

    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing faces with FAN"):
        try:
            face_found = process_image_fan(image_path)
            df.loc[df.index[i], 'face_found'] = face_found
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            df.loc[df.index[i], 'face_found'] = 0

    return df
