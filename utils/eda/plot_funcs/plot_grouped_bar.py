import pandas as pd
import matplotlib.pyplot as plt

def plot_grouped_bar(df, primary_column, secondary_column, primary_labels, secondary_labels):
    """
    Plots a grouped bar chart comparing two columns within the data.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        primary_column (str): The column name for the primary grouping (e.g., age_group).
        secondary_column (str): The column name for the secondary grouping (e.g., gender, race).
        primary_labels (dict): A dictionary mapping primary group values (e.g., age groups) to labels.
        secondary_labels (dict): A dictionary mapping secondary group values (e.g., gender, race) to labels.
    
    Returns:
        None
    """
    prop_df = pd.crosstab(df[primary_column], df[secondary_column], normalize='index')
    
    prop_df.plot(kind='bar', figsize=(10, 6), width=0.8)
    
    plt.title(f'{secondary_column.capitalize()} Distribution per {primary_column.capitalize()}')
    plt.ylabel('Proportion')
    plt.xlabel(f'{primary_column.capitalize()}')
    plt.xticks(ticks=range(len(primary_labels)), labels=[primary_labels.get(i, i) for i in range(len(primary_labels))], rotation=30)
    
    plt.legend(title=secondary_column.capitalize(), labels=[secondary_labels.get(col, col) for col in prop_df.columns])
    
    plt.show()