from PIL import Image
import os
from tqdm import tqdm

def check_image_file_accessibility(folder_path, df_labels):
    """
    Check if image files listed in the DataFrame are valid and can be opened successfully.
    
    Args:
        folder_path (str): The path to the folder containing the image files.
        df_labels (pd.DataFrame): The DataFrame containing the 'filename' column.
    
    Returns:
        dict: A dictionary containing lists of 'valid' and 'invalid' image files.
    """
    valid_files = []
    invalid_files = []

    for file_name in tqdm(df_labels['filename'], desc="Checking image files"):
        file_path = os.path.join(folder_path, file_name)

        try:
            with Image.open(file_path) as img:
                img.verify()
            valid_files.append(file_name)
        except Exception as e:
            invalid_files.append(file_name)
    
    return {
        'valid': valid_files,
        'invalid': invalid_files
    }