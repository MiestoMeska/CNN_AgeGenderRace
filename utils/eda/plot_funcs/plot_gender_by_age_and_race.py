import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_gender_by_age_and_race(df, age_group_column, gender_column, race_column, gender_labels, race_labels, age_group_labels):
    """
    Plot separate bar plots for each race, showing the gender distribution for each age group, 
    arranged in a 3-wide grid layout (two rows with the last empty spot).
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        age_group_column (str): The column name for the age group data.
        gender_column (str): The column name for the gender data.
        race_column (str): The column name for the race data.
        gender_labels (dict): Dictionary mapping gender numeric values to labels.
        race_labels (dict): Dictionary mapping race numeric values to labels.
        age_group_labels (dict): Dictionary mapping age group numeric values to textual labels.
    
    Returns:
        None
    """
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    axes = axes.flatten()
    
    for idx, (race_value, race_label) in enumerate(race_labels.items()):
        race_df = df[df[race_column] == race_value].copy()
        
        race_df.loc[:, 'gender_mapped'] = race_df[gender_column].map(gender_labels)

        sns.countplot(data=race_df, x=age_group_column, hue='gender_mapped', palette='Set2', ax=axes[idx])
        
        axes[idx].set_title(f'Gender Distribution in Each Age Group for {race_label}')
        axes[idx].set_xlabel('Age Group')
        axes[idx].set_ylabel('Count')
        axes[idx].legend(title='Gender')

        axes[idx].set_xticks(range(len(age_group_labels)))
        axes[idx].set_xticklabels([age_group_labels[i] for i in range(len(age_group_labels))], rotation=30)
    
    if len(race_labels) < 6:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.show()