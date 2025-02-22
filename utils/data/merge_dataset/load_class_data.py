import os
import pandas as pd

def load_class_data(folder_path):
    data = []

    for filename in os.listdir(folder_path):
        if filename.startswith("face") and filename.endswith(".jpg"):
            parts = filename.replace(".jpg", "").split('_')

            if len(parts) == 4:
                gender = int(parts[1])
                race = int(parts[2])
                age_group = int(parts[3])

                data.append([filename, gender, race, age_group])

    df = pd.DataFrame(data, columns=['filename', 'gender', 'race', 'age_group'])

    return df