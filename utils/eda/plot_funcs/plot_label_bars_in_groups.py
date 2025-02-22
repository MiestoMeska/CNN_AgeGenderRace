import seaborn as sns
import matplotlib.pyplot as plt

def plot_label_bars_in_groups(df, group_column, label_column, age_group_labels, secondary_labels):
    """
    Plot a bar plot showing the count of each label in each group, with custom group labels.
    
    This function helps visualize the distribution of labels across different age or race groups. The plot includes 
    customized labels for the x-axis is to represent group , and label is represented with different 
    colors for easy comparison.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        group_column (str): The column name for the age group or race data.
        label_column (str): The column name for the label data.
        secondary_labels (dict): Dictionary mapping label numeric values to labels.
        age_group_labels (dict): Dictionary mapping age group or race numeric values to textual labels.
    
    Returns:
        None: The function directly displays the bar plot.
    """

    df['label_mapped'] = df[label_column].map(secondary_labels)


    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=group_column, hue='label_mapped', palette='Set2')
    
    plt.title('Distribution of Labels in Each Group')
    plt.xlabel('Group')
    plt.ylabel('Count')
    plt.legend(title='Label')
    
    plt.xticks(ticks=range(len(age_group_labels)), labels=[age_group_labels[i] for i in range(len(age_group_labels))], rotation=30)
    
    plt.show()