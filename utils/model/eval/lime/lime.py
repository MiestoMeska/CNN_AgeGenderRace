import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from skimage.segmentation import mark_boundaries
from lime import lime_image
from PIL import Image
import random
from functools import partial

class LimeImageDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]['filename'])
        image = Image.open(img_path).convert("RGB")
        label = self.df.iloc[idx][['gender', 'race', 'age_group']].values.astype('float')
        return np.array(image), label

def lime_predict(images, model):
    gender_probs = []

    for image in images:
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(model.device)
        gender_pred, _, _ = model(image_tensor)
        gender_probs.append(torch.softmax(gender_pred, dim=1).cpu().detach().numpy())

    return np.array(gender_probs).squeeze(1)

def find_misclassified_samples(model, dataset, device, classification_head, limit=9):
    misclassified = []
    actual_labels = []
    predicted_labels = []
    images = []

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    model.eval()
    with torch.no_grad():
        for idx in indices:
            if len(misclassified) >= limit:
                break

            image, actual_label = dataset[idx]
            image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            image_tensor = image_tensor.to(device)

            gender_pred, race_pred, age_pred = model(image_tensor)

            if classification_head == "gender":
                actual = int(actual_label[0])
                predicted = torch.argmax(torch.softmax(gender_pred, dim=1)).item()
            elif classification_head == "race":
                actual = int(actual_label[1])
                predicted = torch.argmax(torch.softmax(race_pred, dim=1)).item()
            elif classification_head == "age":
                actual = int(actual_label[2])
                predicted = model.predict_age_group(age_pred).item()

            if actual != predicted:
                misclassified.append(idx)
                actual_labels.append(actual)
                predicted_labels.append(predicted)
                images.append(image)

    return images, actual_labels, predicted_labels


def visualize_misclassified_with_lime(model, dataset, device, classification_head, label_names, grid_size=(5, 2)):
    explainer = lime_image.LimeImageExplainer()

    images, actual_labels, predicted_labels = find_misclassified_samples(
        model, dataset, device, classification_head, limit=grid_size[0]
    )

    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 20))
    num_samples = len(images)

    for i, (image, actual, predicted) in enumerate(zip(images, actual_labels, predicted_labels)):
        if i >= num_samples:
            break

        predict_fn = partial(lime_predict, model=model)
        explanation = explainer.explain_instance(
            image, predict_fn, top_labels=1, hide_color=0, num_samples=20
        )
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(label=top_label, positive_only=False, num_features=10, hide_rest=False)

        row = i
        col_overlay = 0
        col_boundaries = 1

        # Plot original image
        axs[row, col_overlay].imshow(image / 255.0)
        axs[row, col_overlay].set_title(
            f"Original\nActual: {label_names[actual]}\nPredicted: {label_names[predicted]}"
        )
        axs[row, col_overlay].axis("off")

        # Plot LIME boundaries overlay
        axs[row, col_boundaries].imshow(mark_boundaries(temp / 255.0, mask))
        axs[row, col_boundaries].set_title(
            f"LIME Boundaries\nActual: {label_names[actual]}\nPredicted: {label_names[predicted]}"
        )
        axs[row, col_boundaries].axis("off")

    for j in range(num_samples, grid_size[0]):
        axs[j, 0].axis("off")
        axs[j, 1].axis("off")

    plt.tight_layout()
    plt.show()