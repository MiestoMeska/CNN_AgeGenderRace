from facenet_pytorch import MTCNN
import os
import torch
from PIL import Image
from tqdm import tqdm

def process_image_mtcnn(image_path, mtcnn):
    """
    Helper function to process an image and detect faces using MTCNN.
    
    Args:
        image_path (str): Path to the image file.
        mtcnn (MTCNN): The MTCNN model instance for face detection.
    
    Returns:
        int: The number of faces found, or 0 if no face is detected.
    """
    try:
        img = Image.open(image_path)
        boxes, _ = mtcnn.detect(img)

        if boxes is not None and len(boxes) > 0:
            return len(boxes)
        else:
            return 0
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0

def capture_faces_mtcnn(folder_path, df, device=None):
    """
    Detect and mark faces in images using MTCNN with GPU support via PyTorch.
    
    Args:
        folder_path (str): The path to the folder containing the image files.
        df (pd.DataFrame): The DataFrame containing the 'filename' column.
    
    Returns:
        pd.DataFrame: Updated DataFrame with a 'face_found_mtcnn' column indicating the number of faces found.
    """
    if device is None:
        device = "cpu"
    
    mtcnn = MTCNN(keep_all=True, device=device)

    df = df.copy()

    df['face_found_mtcnn'] = 0

    image_paths = [os.path.join(folder_path, file_name) for file_name in df['filename']]

    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing faces with MTCNN"):
        try:
            face_found = process_image_mtcnn(image_path, mtcnn)
            df.loc[df.index[i], 'face_found_mtcnn'] = face_found
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            df.loc[df.index[i], 'face_found_mtcnn'] = 0

    return df