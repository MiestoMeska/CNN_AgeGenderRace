import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from utils import race_labels_FairFace, age_group_labels_Fairface 
#from utils import age_group_labels_Fairface
#from utils import race_labels_Fairface
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize


def evaluate_model(model, test_loader, device):
    model.eval()
    gender_labels = []
    race_labels = []
    age_labels = []
    
    gender_preds = []
    race_preds = []
    age_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            images, labels_gender, labels_race, labels_age = batch

            images = images.to(device)
            labels_gender = labels_gender.to(device)
            labels_race = labels_race.to(device)
            labels_age = labels_age.to(device)
            
            gender_output, race_output, age_output = model(images)

            predicted_age_groups = model.predict_age_group(age_output)

            gender_labels.extend(labels_gender.cpu().numpy())
            race_labels.extend(labels_race.cpu().numpy())
            age_labels.extend(labels_age.cpu().numpy())
            
            gender_preds.extend(torch.softmax(gender_output, dim=1).cpu().numpy())
            race_preds.extend(torch.softmax(race_output, dim=1).cpu().numpy())
            age_preds.extend(predicted_age_groups.cpu().detach().numpy())
    
    gender_preds = np.array(gender_preds)
    race_preds = np.array(race_preds)
    age_preds = np.array(age_preds)

    gender_pred_labels = np.argmax(gender_preds, axis=1)
    race_pred_labels = np.argmax(race_preds, axis=1)
    
    print("\nGender Classification Metrics:")
    print(classification_report(gender_labels, gender_pred_labels, target_names=["Female", "Male"]))
    plot_roc_auc(gender_labels, gender_preds[:, 1], "Gender")

    print("\nRace Classification Metrics:")
    unique_labels = sorted(set(race_labels))
    filtered_race_names = [race_labels_FairFace[label] for label in unique_labels]
    print(classification_report(race_labels, race_pred_labels, target_names=filtered_race_names))
    n_classes_race = len(unique_labels)
    plot_roc_auc_multiclass(race_labels, race_preds, n_classes_race, filtered_race_names)

    print("\nAge Group Classification Metrics:")
    print(classification_report(age_labels, age_preds, target_names=list(age_group_labels_Fairface.values())))
    n_groups_age = 9
    plot_roc_auc_age_groups(age_labels, age_preds, n_groups_age, list(age_group_labels_Fairface.values()))


def plot_roc_auc_age_groups(true_labels, pred_labels, n_groups, class_names):
    true_labels_binarized = label_binarize(true_labels, classes=list(range(n_groups)))

    plt.figure(figsize=(8, 6))
    for i in range(n_groups):
        fpr, tpr, _ = roc_curve(true_labels_binarized[:, i], (pred_labels == i).astype(int))
        roc_auc = roc_auc_score(true_labels_binarized[:, i], (pred_labels == i).astype(int))

        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Age Groups')
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_auc_multiclass(true_labels, pred_probs, n_classes, class_names):
    true_labels_binarized = label_binarize(true_labels, classes=list(range(n_classes)))

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(true_labels_binarized[:, i], pred_probs[:, i])
        roc_auc = roc_auc_score(true_labels_binarized[:, i], pred_probs[:, i])

        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Race Classes')
    plt.legend(loc="lower right")
    plt.show()

def plot_roc_auc(true_labels, pred_probs, title):
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = roc_auc_score(true_labels, pred_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title}")
    plt.legend(loc="lower right")
    plt.show()
