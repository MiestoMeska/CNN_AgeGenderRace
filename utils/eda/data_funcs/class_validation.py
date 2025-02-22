import os
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def validate_and_display_images(folder_path):
    """
    Validate the file names in the specified folder, check for invalid files, and display invalid files in a 3-wide layout.
    
    Args:
        folder_path (str): The path to the folder containing the images.
    
    Returns:
        None
    """
    
    # Regular expression pattern to match the expected file name format
    pattern = re.compile(r'^\d+_[01]_[0-4]_\d{14}')

    invalid_files = []
    chip_files = 0
    total_files = 0

    for file_name in os.listdir(folder_path):
        total_files += 1

        file_base_name = re.sub(r'\.(jpg|chip\.jpg)$', '', file_name)

        if not pattern.match(file_base_name):
            invalid_files.append(file_name)

        if file_name.endswith('.chip.jpg'):
            chip_files += 1

    print(f"Total number of files: {total_files}")
    print(f"Number of preprocessed '.chip.jpg' files: {chip_files}")
    
    if invalid_files:
        print(f"Number of files with invalid format: {len(invalid_files)}")
        print("Invalid files:", invalid_files)
        
        display_invalid_files(folder_path, invalid_files)
    else:
        print("All files have valid class labels.")

def display_invalid_files(folder_path, invalid_files):
    """
    Display the invalid files in a 3-wide layout with multiple rows if necessary.
    
    Args:
        folder_path (str): The path to the folder containing the images.
        invalid_files (list): A list of invalid file names.
    
    Returns:
        None
    """
    num_files = len(invalid_files)
    if num_files > 0:
        num_rows = (num_files + 2) // 3

        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for i, ax in enumerate(axes):
            if i < num_files:
                file_name = invalid_files[i]
                image_path = os.path.join(folder_path, file_name)
                img = Image.open(image_path)

                ax.imshow(img)
                ax.set_title(file_name)
            else:
                ax.axis('off')

            ax.axis('off')

        plt.tight_layout()
        plt.show()