import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

def compute_class_weights(df):
    class_weights_gender = compute_class_weight('balanced', classes=np.array([0, 1]), y=df['gender'])
    class_weights_gender = torch.tensor(class_weights_gender, dtype=torch.float)
    
    class_weights_race = compute_class_weight('balanced', classes=np.array([0, 1, 2, 3, 4]), y=df['race'])
    class_weights_race = torch.tensor(class_weights_race, dtype=torch.float)

    class_weights_age = compute_class_weight('balanced', classes=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]), y=df['age_group'])
    class_weights_age = torch.tensor(class_weights_age, dtype=torch.float)

    return {
        'gender': class_weights_gender,
        'race': class_weights_race,
        'age': class_weights_age
    }