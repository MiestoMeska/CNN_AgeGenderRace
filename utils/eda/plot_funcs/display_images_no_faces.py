import matplotlib.pyplot as plt
import os
from PIL import Image

def display_images_no_faces(folder_path, df, column_name=None, images_per_row=6):
    """
    Display images where no face or more than one face was found in the specified column.
    
    Args:
        folder_path (str): The path to the folder containing the image files.
        df (pd.DataFrame): The DataFrame containing 'filename' and the specified column.
        column_name (str): The name of the column containing the face count data.
        images_per_row (int): Number of images to display per row. Default is 6.
    
    Returns:
        None
    """

    if column_name is None:
        print("No column name was provided. Exiting..")
        return

    # Filter images with no face or more than 1 face found
    no_faces_df = df[(df[column_name] == 0) | (df[column_name] > 1)]
    no_face_filenames = no_faces_df['filename'].values
    no_face_values = no_faces_df[column_name].values

    num_images = len(no_face_filenames)
    num_rows = (num_images + images_per_row - 1) // images_per_row 

    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(15, 2 * num_rows))
    axes = axes.flatten()

    for i, file_name in enumerate(no_face_filenames):
        file_path = os.path.join(folder_path, file_name)
        
        try:
            img = Image.open(file_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"{no_face_values[i]}", fontsize=10)
        except Exception as e:
            print(f"Error displaying image {file_name}: {e}")
            axes[i].axis('off')
    
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.tight_layout()
    plt.show()