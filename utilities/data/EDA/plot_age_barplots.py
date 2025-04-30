import seaborn as sns
import matplotlib.pyplot as plt

def plot_age_barplots(df, age_column, gender_column, race_column, gender_labels, race_labels):
    """
    Plot barplots for the age distribution, age with gender, and age with race, with small gaps between bars.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        age_column (str): The column name for the age data.
        gender_column (str): The column name for the gender data.
        race_column (str): The column name for the race data.
        gender_labels (dict): Dictionary mapping gender numeric values to labels.
        race_labels (dict): Dictionary mapping race numeric values to labels.
    
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[age_column], kde=False, bins=20, color='skyblue', shrink=0.9)
    plt.title('Age Distribution (Overall)')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(10, 6))
    
    temp_gender_column = df[gender_column].map(gender_labels)
    
    gender_plot = sns.histplot(data=df, x=age_column, hue=temp_gender_column, multiple='dodge', bins=20, shrink=0.9)
    plt.title('Age Distribution by Gender')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

    handles, labels = gender_plot.get_legend_handles_labels()
    if len(handles) > 0:
        gender_plot.legend(handles, labels, title='Gender')

    plt.show()

    plt.figure(figsize=(10, 6))
    
    temp_race_column = df[race_column].map(race_labels)
    
    race_plot = sns.histplot(data=df, x=age_column, hue=temp_race_column, multiple='dodge', bins=20, shrink=0.9)
    plt.title('Age Distribution by Race')
    plt.xlabel('Age')
    plt.ylabel('Frequency')

    handles, labels = race_plot.get_legend_handles_labels()
    if len(handles) > 0:
        race_plot.legend(handles, labels, title='Race')

    plt.show()