import os
import pandas as pd

def get_df_filenames(output_folder):
    """
    Create a DataFrame from the image filenames in the specified folder.

    Args:
        output_folder (str): Path to the folder containing image files.
    
    Returns:
        pd.DataFrame: DataFrame containing the filenames of images.
    """
    image_files = [f for f in os.listdir(output_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    df_labels = pd.DataFrame(image_files, columns=['filename'])
    
    return df_labels