import os
import re
import pandas as pd
import numpy as np

def extract_class_data(folder_path):
    """
    Extract class data from filenames and create a pandas DataFrame with columns for filename, age, gender, and race.
    If a filename does not follow the expected pattern, fill age, gender, and race with 'NaN'.
    
    Args:
        folder_path (str): The path to the folder containing the images.
    
    Returns:
        pd.DataFrame: A DataFrame with columns: 'filename', 'age', 'gender', 'race'.
    """
    pattern = re.compile(r'^(\d+)_(\d)_(\d)_(\d{14})')

    data = []

    for file_name in os.listdir(folder_path):
        file_base_name = re.sub(r'\.(jpg|chip\.jpg)$', '', file_name)
        match = pattern.match(file_base_name)

        if match:
            age, gender, race, _ = match.groups()
            data.append([file_name, age, gender, race])
        else:
            data.append([file_name, np.nan, np.nan, np.nan])

    df = pd.DataFrame(data, columns=['filename', 'age', 'gender', 'race'])
    
    return df