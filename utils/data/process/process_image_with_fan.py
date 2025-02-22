import torch
import face_alignment
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import shutil

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector='sfd', device='cuda' if torch.cuda.is_available() else 'cpu')

def show_image(image, title="Image after step"):

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    print(f"Image Dimensions: {width}x{height}, Channels: {channels}")
    
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    plt.show()

def draw_landmarks(image, landmarks, title="Landmarks"):
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
    show_image(image, title)

def resize_image(image):

    h, w = image.shape[:2]
    
    if max(h, w) > 2048:
        if h > w:
            new_h = 2048
            new_w = int(w * (2048 / h))
        else:
            new_w = 2048
            new_h = int(h * (2048 / w))
        resized_image = cv2.resize(image, (new_w, new_h))
    
    elif max(h, w) < 256:
        if h > w:
            new_h = 256
            new_w = int(w * (256 / h))
        else:
            new_w = 256
            new_h = int(h * (256 / w))
        resized_image = cv2.resize(image, (new_w, new_h))
    
    else:
        resized_image = image
    
    return resized_image

def crop_face(image, landmarks, target_size=(200, 200), bbox_height_ratio=0.80, eye_midpoint_top_margin_ratio=0.72):
    """
    Crop and align the face from the image using facial landmarks.
    Ensure the eye midpoint is always centered horizontally, and 57 px from the top of the cropped image.
    
    Args:
        image (np.array): The original image.
        landmarks (np.array): Facial landmarks (68 points).
        target_size (tuple): The desired size of the output image.
        desired_eye_to_chin_ratio (float): The desired ratio of the eye-chin distance to the image height.
        eye_midpoint_top_margin (int): The vertical distance between the eye midpoint and the top of the output image.
    
    Returns:
        np.array: Cropped and aligned face image, or None if the face is out of bounds.
    """
    left_eye = landmarks[36]
    right_eye = landmarks[45]
    chin = landmarks[8]

    eye_midpoint = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    dX = chin[0] - eye_midpoint[0]
    dY = chin[1] - eye_midpoint[1]
    angle = np.degrees(np.arctan2(dY, dX)) - 90 

    desired_bbox_heigh = bbox_height_ratio * target_size[0]
    current_bbox_height = max(landmarks[:, 1]) - min(landmarks[:, 1])
    scale_factor = desired_bbox_heigh / current_bbox_height

    M = cv2.getRotationMatrix2D(eye_midpoint, angle, scale_factor)
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    ones = np.ones((landmarks.shape[0], 1))
    landmarks_homo = np.hstack([landmarks, ones])
    rotated_landmarks = M.dot(landmarks_homo.T).T

    x_min = int(min(rotated_landmarks[:, 0]))
    x_max = int(max(rotated_landmarks[:, 0]))
    y_min = int(min(rotated_landmarks[:, 1]))
    y_max = int(max(rotated_landmarks[:, 1]))

    if x_min < 0 or y_min < 0 or x_max > rotated_image.shape[1] or y_max > rotated_image.shape[0]:
        #print("Face out of bounds. Skipping this face.")
        return None
    
    bbox_height = y_max - y_min
    if bbox_height < 100:
        #print("Face bounding box too small (less than 100px). Skipping this face.")
        return None

    midpoint_x = eye_midpoint[0]
    midpoint_y = eye_midpoint[1]

    crop_x_min = max(0, int(midpoint_x - target_size[0] // 2))
    crop_x_max = min(rotated_image.shape[1], int(midpoint_x + target_size[0] // 2))

    crop_y_min = max(0, int(midpoint_y - target_size[0]* 0.3))
    crop_y_max = min(rotated_image.shape[0], int(crop_y_min + target_size[1]))

    cropped_image = rotated_image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    cropped_h, cropped_w = cropped_image.shape[:2]

    pad_top = (target_size[1] - cropped_h) // 2 if cropped_h < target_size[1] else 0
    pad_bottom = target_size[1] - cropped_h - pad_top if cropped_h < target_size[1] else 0
    pad_left = (target_size[0] - cropped_w) // 2 if cropped_w < target_size[0] else 0
    pad_right = target_size[0] - cropped_w - pad_left if cropped_w < target_size[0] else 0

    padded_image = cv2.copyMakeBorder(
        cropped_image,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    return padded_image


def process_image_with_fan(image_path, target_size=(200, 200)):

    if not os.path.exists(image_path):
        #print(f"Image path {image_path} does not exist.")
        return None

    original_image = cv2.imread(image_path)
    if original_image is None:
        #print(f"Failed to load image at {image_path}")
        return None

    if isinstance(original_image, list):
       #print(f"Image is not valid: {image_path}")
        return None
    

    resized_image = resize_image(original_image)
    #show_image(resized_image, title="resized image")
    preds = fa.get_landmarks(resized_image)
    faces = []

    if preds is None or len(preds) == 0:
        print("No faces detected.")

    else:
        #print(f"{len(preds)} face(s) detected.")


        for landmarks in preds:
            face_image = crop_face(resized_image, landmarks=landmarks)

            if face_image is not None:
                faces.append(face_image)
                #show_image(face_image, title="Image after rec")



    if faces is None or len(faces) == 0:
        #print(f"No face detected in {image_path}")
        return None
    
    return faces

def process_images(input_folder, output_folder, target_size=(200, 200), num_images=None):
    """
    Process a specified number of image files from the input folder using FAN and save the processed images to the output folder.
    
    Args:
        input_folder (str): Path to the folder containing the images.
        output_folder (str): Path to the folder where processed images will be saved.
        target_size (tuple): Desired output image size (default is 200x200).
        num_images (int or None): Number of randomly selected images to preprocess. If None, processes all images.
    """
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist.")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if num_images is not None:
        image_files = random.sample(image_files, min(num_images, len(image_files)))

    for filename in image_files:
        input_file_path = os.path.join(input_folder, filename)
        
        processed_faces = process_image_with_fan(input_file_path, target_size)

        if processed_faces is not None:
            for i, face in enumerate(processed_faces):
                output_file_path = os.path.join(output_folder, f"processed_{i}_{filename}")
                
                cv2.imwrite(output_file_path, face)
        else:
            print(f"Face could not be processed in file: {filename}")

