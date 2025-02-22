import os
import numpy as np
import pandas as pd
import cv2

def batch_loader(df, filenames, image_dir, batch_size=100, image_size=(64, 64)):
    """
    Generator function to load images in batches from the directory.
    
    Args:
    - filenames: list of image filenames.
    - image_dir: directory where the images are stored.
    - batch_size: number of images to load in a single batch.
    - image_size: size to which each image will be resized.
    
    Yields:
    - X_batch: numpy array of images in the batch.
    - y_batch_gender: corresponding gender labels.
    - y_batch_race: corresponding race labels.
    - y_batch_age: corresponding age group labels.
    """
    n_samples = len(filenames)
    
    shuffled_indices = np.random.permutation(np.arange(n_samples))
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = shuffled_indices[start_idx:end_idx]
        batch_filenames = filenames[batch_indices]
        
        X_batch = []
        y_batch_gender = []
        y_batch_race = []
        y_batch_age = []
        
        for filename in batch_filenames:
            img_path = os.path.join(image_dir, filename)
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, image_size)
                img = img / 255.0
                X_batch.append(img.flatten())
                
                label = df[df['filename'] == filename]
                y_batch_gender.append(label['gender'].values[0])
                y_batch_race.append(label['race'].values[0])
                y_batch_age.append(label['age_group'].values[0])
        
        X_batch = np.array(X_batch)
        y_batch_gender = np.array(y_batch_gender)
        y_batch_race = np.array(y_batch_race)
        y_batch_age = np.array(y_batch_age)
        
        yield X_batch, y_batch_gender, y_batch_race, y_batch_age