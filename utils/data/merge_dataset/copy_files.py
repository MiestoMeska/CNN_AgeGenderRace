import os
import shutil

def copy_files(df, dest_dir):
    for index, row in df.iterrows():
        file_path = row['file_path']
        gender = row['gender']
        race = row['race']
        age_group = row['age_group']
        
        new_filename = f"face{index}_{gender}_{race}_{age_group}.jpg"
        
        dest_path = os.path.join(dest_dir, new_filename)
        
        try:
            shutil.copy(file_path, dest_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error copying file {file_path}: {e}")