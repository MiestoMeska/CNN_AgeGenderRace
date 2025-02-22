gender_labels_UTK = {
    0: 'Male',
    1: 'Female'
}

race_labels_UTK = {
    0: 'White',
    1: 'Black',
    2: 'Asian',
    3: 'Indian',
    4: 'Others'
}

age_group_labels_UTK = {
    0: "0-2",
    1: "3-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "70+"
}

race_labels_FairFace = {
    0: 'East Asian',
    1: 'Indian',
    2: 'Black',
    3: "White",
    4: "Middle Eastern",
    5: "Latino_Hispanic",
    6: "Southeast Asian",
}

gender_labels_FairFace = {
    0: "Male",
    1: "Female"
}

age_group_labels_Fairface = {
    0: "0-2",
    1: "3-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "70+"
}

def map_fairface_to_utk(race_fairface):
    if race_fairface in [0, 6]:  # 0: 'East Asian', 6: 'Southeast Asian'
        return 2  # Map to 'Asian' in UTKFace
    elif race_fairface == 3:  # 'White'
        return 0  # 'White'
    elif race_fairface == 2:  # 'Black'
        return 1  # 'Black'
    elif race_fairface == 1:  # 'Indian'
        return 3  # 'Indian'
    else:  # 4: 'Middle Eastern', 5: 'Latino_Hispanic'
        return 4  # 'Others'
    
def map_age_to_group(age):
    if 0 <= age <= 2:
        return 0  # '0-2'
    elif 3 <= age <= 9:
        return 1  # '3-9'
    elif 10 <= age <= 19:
        return 2  # '10-19'
    elif 20 <= age <= 29:
        return 3  # '20-29'
    elif 30 <= age <= 39:
        return 4  # '30-39'
    elif 40 <= age <= 49:
        return 5  # '40-49'
    elif 50 <= age <= 59:
        return 6  # '50-59'
    elif 60 <= age <= 69:
        return 7  # '60-69'
    else:
        return 8  # '70+'
