import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_data():
    data = datasets.load_iris()
    return data

def load_df():
    data = datasets.load_iris()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    # df['species'] = data.target_names[data.target] # Add species
    return df

def load_split(test_size=30):
    data = datasets.load_iris()
    return train_test_split(data.data, data.target, test_size=test_size)

def missing():
    return load_df().isnull().sum()