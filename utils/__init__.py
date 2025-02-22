# data labels

from.data.data_label_dict import gender_labels_UTK, race_labels_UTK, age_group_labels_UTK, race_labels_FairFace, gender_labels_FairFace, age_group_labels_Fairface 
from.data.data_label_dict import map_fairface_to_utk,  map_age_to_group

# data functions

from .eda.data_funcs.compare_folders import compare_folders
from .eda.data_funcs.class_validation import validate_and_display_images
from .eda.data_funcs.extract_class_data import extract_class_data
from .eda.data_funcs.get_class_ranges import get_class_ranges
from .eda.data_funcs.check_image_file_accessibility import check_image_file_accessibility
from .eda.data_funcs.capture_faces_mtcnn import capture_faces_mtcnn
from .eda.data_funcs.capture_faces_fan import capture_faces_fan
from .eda.data_funcs.pca import load_images_as_flattened_array
from .eda.data_funcs.pca import plot_principal_components_as_images

# plotting functions

from .eda.plot_funcs.plot_image_dimensions import plot_image_dimensions
from .eda.plot_funcs.display_images_no_faces import display_images_no_faces
from .eda.plot_funcs.plot_pie_chart import plot_pie_chart
from .eda.plot_funcs.plot_age_distribution_line import plot_age_distribution_line
from .eda.plot_funcs.plot_age_barplots import plot_age_barplots
from .eda.plot_funcs.plot_grouped_bar import plot_grouped_bar
from .eda.plot_funcs.plot_label_bars_in_groups import plot_label_bars_in_groups
from .eda.plot_funcs.plot_gender_by_age_and_race import plot_gender_by_age_and_race

# process functions

from .data.process.process_images_with_landmarks import process_images_with_landmarks
from .data.process.process_image_with_fan import process_image_with_fan, process_images
from .data.process.get_df_filenames import get_df_filenames

# landmark analysis

from .data.landmarks_analysis.gather_all_outliers import gather_all_outliers
from .data.landmarks_analysis.gather_all_outliers_in_batches import gather_all_outliers_in_batches
from .data.landmarks_analysis.display_images_for_outliers import display_images_for_outliers
from .data.landmarks_analysis.gather_landmark_data import gather_landmark_data
from .data.landmarks_analysis.plot_multiple_normalized_landmarks import plot_multiple_normalized_landmarks
from .data.landmarks_analysis.plot_multiple_normalized_landmarks_with_outliers import plot_multiple_normalized_landmarks_with_outliers
from .data.landmarks_analysis.plot_eye_distance_distribution import plot_eye_distance_distribution
from .data.landmarks_analysis.plot_distance_distributions_and_bbox_scatter import plot_distance_distributions_and_bbox_scatter
from .data.landmarks_analysis.plot_aspect_ratio_distribution import plot_aspect_ratio_distribution
from .data.landmarks_analysis.plot_face_size_distribution import plot_face_size_distribution
from .data.landmarks_analysis.find_and_plot_outliers import find_and_plot_outliers

# dataset merging

from .data.merge_dataset.gather_data import gather_utkface_data, gather_fairface_data
from .data.merge_dataset.copy_files import copy_files
from .data.merge_dataset.load_class_data import load_class_data

## Models ##

# ML model 

from .model.ML_model_base.batch_loader import batch_loader
from .model.ML_model_base.fit_ipca_with_batches import fit_ipca_with_batches
from .model.ML_model_base.show_explained_variance import show_explained_variance
from .model.ML_model_base.show_principal_components import show_principal_components
from .model.ML_model_base.show_scree_plot import show_scree_plot
from .model.ML_model_base.transform_with_ipca import transform_with_ipca

# CNN model

from .model.MultiLabelImageDataset import MultiLabelImageDataset
from .model.compute_class_weights import compute_class_weights

from .model.CNN_base.ResNetMultiOutputCNN import ResNetMultiOutputCNN
from .model.CNN_resnet.ResNetMultiOutputCNN_extended import ResNetMultiOutputCNN_extended
from .model.CNN_resnet.ResNetMultiOutputCNN_mean import ResNetMultiOutputCNN_mean

from .model.CNN_effnet.EfficientNetMultiOutput import EfficientNetMultiOutput
from .model.CNN_effnet.EfficientNetMultiOutput_ordinalReg import EfficientNetMultiOutput_ordinalReg
from .model.CNN_effnet.EfficientNetMultiOutput_class import EfficientNetMultiOutput_class
from .model.CNN_effnet.EfficientNetMultiOutput_shared import EfficientNetMultiOutput_shared

from .model.UnfreezeCallback import UnfreezeCallback
from .model.UnfreezeCallback_EffNet import UnfreezeCallback_EffNet
from .model.UnfreezeCallback_EffNet_thresholds import UnfreezeCallback_EffNet_thresholds

# EVAL

from .model.eval.evaluate_model import evaluate_model

# LIME

from .model.eval.lime.lime import LimeImageDataset
from .model.eval.lime.lime import lime_predict
from .model.eval.lime.lime import find_misclassified_samples
from .model.eval.lime.lime import visualize_misclassified_with_lime