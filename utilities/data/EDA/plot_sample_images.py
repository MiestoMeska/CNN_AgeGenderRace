import os
import cv2
import matplotlib.pyplot as plt

def plot_sample_images(df_labels, path_data, n_images=64,
                       gender_labels=None, race_labels=None):
    """
    Plots face images with age, gender, and race labels (mapped from dicts).

    Args:
        df_labels (pd.DataFrame): DataFrame with columns: 'filename', 'age', 'gender', 'race'.
        path_data (str): Directory where images are stored.
        n_images (int): Number of images to show.
        gender_labels (dict): Mapping from int to gender string.
        race_labels (dict): Mapping from int to race string.
    """
    if n_images > len(df_labels):
        raise ValueError("n_images is greater than the number of available samples.")

    sampled_df = df_labels.sample(n_images).reset_index(drop=True)
    grid_size = int(n_images ** 0.5)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.ravel()

    for i, row in sampled_df.iterrows():
        image_path = os.path.join(path_data, row['filename'])
        img = cv2.imread(image_path)

        if img is None:
            print(f"Warning: Could not read image {image_path}")
            axes[i].axis('off')
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Correct label mapping here
        age = row['age']
        gender = gender_labels[int(row['gender'])]
        race = race_labels[int(row['race'])]

        axes[i].imshow(img_rgb)
        axes[i].axis('off')
        axes[i].set_title(f"Age: {age} | {gender} | {race}", fontsize=8)

    plt.tight_layout()
    plt.show()
