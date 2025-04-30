import seaborn as sns
import matplotlib.pyplot as plt

def plot_age_distribution_line(df, age_column, secondary_column, secondary_label_dict=None):
    """
    Plot age distribution as lines, categorized by a secondary column (e.g., gender, race).

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        age_column (str): Column name for the age data.
        secondary_column (str): Column name for the secondary categorical data.
        secondary_label_dict (dict): Optional dictionary to map the secondary column values (e.g., gender or race) 
                                     to human-readable labels.
    
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))

    if secondary_label_dict:
        temp_column = df[secondary_column].map(secondary_label_dict)
    else:
        temp_column = df[secondary_column]

    for category in temp_column.unique():
        sns.kdeplot(df[temp_column == category][age_column], label=category, linewidth=2)

    plt.title(f"Age Density Distribution by {secondary_column}")
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend(title=secondary_column)
    plt.grid(True)
    plt.show()