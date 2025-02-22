import os
import re
import pandas as pd
import numpy as np

def gather_utkface_data(folder_path):
    """
    Gathers the file paths, gender, race, and age data from UTKFace dataset.

    Args:
        folder_path (str): Path to the UTKFace image dataset.

    Returns:
        pd.DataFrame: DataFrame containing 'file_path', 'gender', 'race', and 'age'.
    """
    pattern = re.compile(r'^(\d+)_(\d)_(\d)_(\d{14})')
    
    data = []
    
    for file_name in os.listdir(folder_path):
        file_base_name = re.sub(r'\.(jpg|chip\.jpg)$', '', file_name)
        
        match = pattern.match(file_base_name)
        
        if match:
            age, gender, race, _ = match.groups()
            file_path = os.path.join(folder_path, file_name)
            data.append([file_path, int(gender), int(race), int(age)])
        else:
            data.append([file_name, np.nan, np.nan, np.nan])
    
    df = pd.DataFrame(data, columns=['file_path', 'gender', 'race', 'age'])
    
    return df

def gather_fairface_data(folder_path):
    """
    Gathers the file paths, gender, race, and age group data from FairFace dataset.

    Args:
        folder_path (str): Path to the FairFace image dataset.

    Returns:
        pd.DataFrame: DataFrame containing 'file_path', 'gender', 'race', and 'age_group'.
    """

    pattern = re.compile(r'processed_0_\d+_(\d)_(\d)_(\d)_FairFace')
    
    data = []
    
    for file_name in os.listdir(folder_path):
        match = pattern.match(file_name)
        
        if match:
            age_group, gender, race = match.groups()
            file_path = os.path.join(folder_path, file_name)
            data.append([file_path, int(gender), int(race), int(age_group)])

    df = pd.DataFrame(data, columns=['file_path', 'gender', 'race', 'age_group'])
    
    return df