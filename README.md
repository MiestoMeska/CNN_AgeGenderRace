# Practical Deep Learning
# Age, Race and Gender classification

![Faces](https://github.com/TuringCollegeSubmissions/vruzga-DL.3.5/blob/master/assets/img/Faces_generic.JPG)

## Introduction

In this project, we tackle a multi-objective classification task using the UTKFace and FairFace datasets, which contains labeled images of faces. The goal is to develop a single model capable of predicting a person’s age, race, and gender from an image. As with any model dealing with sensitive information such as race and gender, it is important to analyze the model’s performance not only from a technical perspective but also from an ethical one. We will explore potential biases that might arise from the dataset and model design and discuss mitigation strategies.

## Project Task

**Main Goal:** Develop a multi-task image classifier that can predict three attributes from a single image: age, race, and gender. The model will be trained on the merged dataset of UTKFace dataset and FairFace dataset. Evaluation of the model for performance across each task. Additionally, the ethical implications of the model will be investigated.

### Concepts to Explore

In this project, we build on our deep learning knowledge to tackle a complex classification problem using advanced techniques and tools. Our focus will include:

- **Multi-Task Learning:** We will explore multi-task learning to build a model that simultaneously predicts multiple objectives (age, race, and gender) from facial images.

- **Convolutional Neural Networks (CNNs):** CNNs will be employed for feature extraction from images, as they are highly effective at capturing spatial hierarchies in visual data. We will design a CNN-based architecture that handles all three tasks.

- **Loss Functions:** Different loss functions will be combined to optimize the model for multi-task learning. The tasks include classification of gender, race, and age groups.

- **Model Interpretability with LIME:** To ensure transparency, we will use LIME (Local Interpretable Model-Agnostic Explanations) to understand the model’s decision-making process on individual samples. This will help identify which parts of the image contribute to the model’s predictions.

- **AI Ethics and Bias:** We will critically assess the potential biases in the model’s predictions, particularly around race and gender. The model’s misclassifications will be examined to detect any trends, such as biased performance towards certain demographic groups. We will explore potential causes of these biases, such as dataset imbalance, and propose strategies to mitigate them.

## Project Content

### Data

- **Acquisition:** The dataset used for this project is provided [by the UTKFace dataset on Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new). It contains facial images labeled with three attributes: age, race, and gender. The dataset spans a wide range of ages and includes diverse racial and gender labels, making it suitable for multi-objective classification tasks.

- **Dataset Exploration:** [An Exploratory Data Analysis (EDA)](https://github.com/TuringCollegeSubmissions/vruzga-DL.3.5/blob/master/notebooks/1.EDA.ipynb) provides insights into the distribution of age, race, and gender categories within the UTKFace dataset. This analysis helps understand the balance of the dataset, detect any potential class imbalances, and identify key characteristics of the images, such as the age group distributions and racial diversity. Understanding these distributions is essential for guiding model design and ensuring fair representation across all categories.

- **Expansion of the Dataset:** The dataset expansion process is documented across three notebooks, with each step aimed at improving the quality and consistency of the combined dataset:

1. Analyzing Face Positioning in the UTKFace Dataset: [The Face Alignment Exploration Notebook](https://github.com/TuringCollegeSubmissions/vruzga-DL.3.5/blob/master/notebooks/2.face_alignment_exploration.ipynb) analyzes the positioning of faces in the UTKFace dataset. This ensures that the facial regions are consistently aligned, which is crucial for achieving high model performance.
2. Processing Images for Consistent Zoom, Crop, and Face Alignment: [The Image processing Notebook](https://github.com/TuringCollegeSubmissions/vruzga-DL.3.5/blob/master/notebooks/3.image_process_FairFace.ipynb) focuses on processing the FairFace dataset to ensure the same zoom level, crop, and face alignment as UTKFace. This preprocessing step standardizes the images across datasets, making them compatible for merging.
3. Label Alignment and Merging Datasets: [The Dataset Merging Notebook](https://github.com/TuringCollegeSubmissions/vruzga-DL.3.5/blob/master/notebooks/4.dataset_merging.ipynb) deals with aligning the labels between the UTKFace and FairFace datasets and merging them into a unified dataset. This ensures that the combined dataset has consistent and properly formatted labels for age, race, and gender.

- **The Expanded Dataset EDA:** After merging the UTKFace and FairFace datasets, the combined dataset was cleaned by retaining only images where faces were detected using the MTCNN and FAN models. The data cleaning process is documented in the [The Merged Dataset Cleaning Notebook](https://github.com/TuringCollegeSubmissions/vruzga-DL.3.5/blob/master/notebooks/5.merged_dataset_cleaning.ipynb).


Following the cleaning, an exploratory data analysis (EDA) was conducted on the cleaned dataset. The entire process and key insights from the EDA are also detailed in the [The Merged Dataset EDA Notebook](https://github.com/TuringCollegeSubmissions/vruzga-DL.3.5/blob/master/notebooks/6.EDA_merged_dataset.ipynb).

### Model Creation

- **Creating the Baseline Model:** 

The first step in creating a model for the gender, race, and age classification task was to establish a strong baseline. Principal Component Analysis (PCA) was applied to the image data to reduce dimensionality while retaining key features. Using the principal components, two baseline models were trained:

1. Random Forest Classifier: A Weighted Random Forest Classifier was trained on the principal components to predict gender, race, and age. Class weights were applied to address imbalances in the dataset, ensuring fair representation across categories. The model served as a quick, interpretable baseline to assess the separability of the classes based on the extracted features.

2. XGBoost Classifier: To complement the Random Forest model, an XGBoost Classifier was also trained on the principal components. XGBoost's ability to handle imbalanced data and capture non-linear relationships provided alternative for the baseline performance evaluation.

These models offered valuable insights into the dataset's characteristics and set a reference point for evaluating the performance of more complex deep learning architectures.

The process of creating baseline model is provided in the [Baseline ML model creation Notebook.](https://github.com/TuringCollegeSubmissions/vruzga-DL.3.5/blob/master/notebooks/7.ML_model_train.ipynb)

- **CNN Model Creation Routine:**

A series of Convolutional Neural Network (CNN) models were developed and trained for gender, race, and age classification. The training routine involved experimenting with different architectures and hyperparameters to achieve optimal performance across all tasks.

The CNN model experimentation was based on the EfficientNet pre-trained architecture. The approach included:

* Adding a shared layer for feature extraction followed by three separate heads for gender, race, and age classification tasks.
* Experimenting with the complexity of the shared and head layers, including varying the number of layers and neurons.
* Testing different activation functions, such as ReLU, Leaky ReLU, and sigmoid, to optimize performance for each classification task.
* The routine for training models is documented in the [Model Training Notebook.](https://github.com/TuringCollegeSubmissions/vruzga-DL.3.5/blob/master/notebooks/8.CNN_train.ipynb)

All model metrics were tracked using TensorBoard, allowing for real-time monitoring of training and validation performance. This facilitated the selection of the best-performing model based on comprehensive performance comparisons.

![Metrics](https://github.com/TuringCollegeSubmissions/vruzga-DL.3.5/blob/master/assets/img/train_metrics.JPG)

- **Final Model Evaluation:**

The trained multi-output EfficientNet-based model shows promising results in gender, race, and age group classification.

**Pros**

* High Gender Classification Accuracy: Achieved 86% accuracy with an AUC of 0.94, showing strong performance in gender prediction.
* Good Performance on Some Race Classes: Performed well on Indian and Black classes (F1-score: 0.76, AUC > 0.93).
* Effective Multi-Task Learning: Handles gender, race, and age classification in a single architecture.
* Accurate for Young Age Groups: Highest age group AUC (0.87) for the 0–2 category.

**Cons**

* Moderate Race Classification: Overall race accuracy at 68%, with lower performance on Middle Eastern (F1-score: 0.54).
* Weak Age Prediction: 46% accuracy for age groups, with AUC scores 0.61–0.70 for older categories.
* Class Imbalance Sensitivity: Struggles with underrepresented classes, affecting generalization.

Detailed evaluation and visualizations ar provided in [Model Evaluation Notebook.](https://github.com/TuringCollegeSubmissions/vruzga-DL.3.5/blob/master/notebooks/9.model_eval.ipynb)


## Conclusions of the Project

The multi-output EfficientNet-based model was evaluated for gender, race, and age group classification tasks. The evaluation included performance metrics, ROC curve analysis, and interpretability through LIME visualizations.

The model performed strongly in gender classification, achieving an 86% accuracy with an AUC of 0.94, indicating reliable discrimination between male and female classes. For race classification, the overall accuracy was 68%, with better performance on the Indian and Black categories but lower accuracy for the Middle Eastern class. The age classification task proved the most challenging, with an accuracy of 46% and significant misclassifications, especially for older age groups.

LIME analysis provided interpretability by highlighting regions the model focused on for its predictions. For gender, relevant facial features were considered, but some misclassifications were linked to ambiguous or mislabeled data. In race classification, the model consistently misclassified various races as East Asian, revealing a potential dataset imbalance. In age classification, LIME showed that the model struggled to distinguish adjacent age groups, often misclassifying older individuals as significantly younger, likely due to insufficient age-related training data.

 **Key Conclusions**

✅ Gender classification shows robust performance with high accuracy and balanced precision-recall across classes.  
⚠️ Race classification reveals biases, with a tendency to overpredict certain classes (e.g., East Asian), suggesting class imbalance issues.  
⚠️ Age classification struggles the most, particularly with older age groups, due to overlapping age features and lack of clear age boundaries in the dataset.  
🧩 LIME interpretability confirms that the model relies on meaningful facial regions but is sometimes misled by irrelevant areas or noise in the data.  
⚡ Label inconsistencies and ambiguous samples contribute significantly to misclassifications, indicating a need for refined data quality.  

**Suggestions for Future Improvements**

1. Data Quality Enhancement:

Correct mislabeled samples and remove ambiguous data entries.
Balance the dataset by augmenting underrepresented race and age groups.

2. Advanced Model Techniques:

Ordinal loss functions for age prediction to better handle the continuous nature of aging.
Region-focused attention mechanisms to reduce the influence of irrelevant image regions.

3. Improved Feature Extraction:

Improve facial landmark-based preprocessing to ensure uniform focus on key facial areas.
Experiment with deeper EfficientNet variants or hybrid architectures to capture more nuanced age and race features.

4. Bias Mitigation Strategies:

Implement fairness-aware training to reduce racial and gender bias.
Apply domain adaptation techniques if the dataset contains cross-domain variations (e.g., ethnicity, lighting conditions).

5. Explainability & Interpretability:

Further use LIME to continuously monitor and interpret model predictions, ensuring transparency in decision-making.

**Final Thoughts**

This project demonstrated the feasibility of using a multi-task EfficientNet-based model for simultaneous prediction of gender, race, and age group. While gender classification achieved satisfactory performance, race and age predictions require further enhancements. Addressing data imbalance, refining the model architecture, and introducing fairness-focused strategies will be crucial for improving classification accuracy and ensuring possible ethical deployment in real-world applications.