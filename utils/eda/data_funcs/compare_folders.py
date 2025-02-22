import os
import hashlib
from tqdm import tqdm

def get_files(folder_path):
    """
    Get the set of relative file paths from a folder.

    Args:
        folder_path (str): The root directory to search for files.

    Returns:
        set: A set of relative file paths found within the folder.
    """
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.relpath(os.path.join(root, file), folder_path))
    return set(file_paths)

def sha256_hash(filepath):
    """
    Calculate the SHA256 hash of a file.

    Args:
        filepath (str): Path to the file to hash.

    Returns:
        str: The SHA256 hash as a hexadecimal string.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def compare_folders(folder1, folder2):
    """
    Compare the files in two folders by file names and content (using SHA256 hash).

    Args:
        folder1 (str): The path to the first folder.
        folder2 (str): The path to the second folder.

    Returns:
        tuple: Two lists of unique files (files only found in folder1 and files only found in folder2).
    """
    folder1_files = get_files(folder1)
    folder2_files = get_files(folder2)

    unique_in_folder1 = []
    unique_in_folder2 = []

    if folder1_files == folder2_files:
        print("File names are identical in both folders.")
    else:
        print("File names differ between the folders.")
        only_in_folder1 = folder1_files - folder2_files
        only_in_folder2 = folder2_files - folder1_files
        unique_in_folder1.extend(only_in_folder1)
        unique_in_folder2.extend(only_in_folder2)

    identical_count = 0
    different_count = 0
    common_files = folder1_files.intersection(folder2_files)

    for file in tqdm(common_files, desc="Comparing files", unit="file"):
        file1_path = os.path.join(folder1, file)
        file2_path = os.path.join(folder2, file)
        if sha256_hash(file1_path) == sha256_hash(file2_path):
            identical_count += 1
        else:
            different_count += 1

    print(f"\nNumber of identical files: {identical_count}")
    print(f"Number of unique files: {different_count}")
    print(f"Number of unique files in {folder1}: {len(unique_in_folder1)}")
    print(f"Number of unique files in {folder2}: {len(unique_in_folder2)}")

    return unique_in_folder1, unique_in_folder2

