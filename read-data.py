import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

def print_basic_summary(df):
    # Print data
    print("First lines of data: ")
    print(df.head())

    # Print data types
    print("Data types: ")
    print(df.dtypes)

    # Print data shape
    print("Data shape: ")
    print(df.shape)

    #Print summary
    print("Summary: ")
    print(df.describe())

    # Print number of missing values
    print("Number of missing values: ")
    print(df.isnull().sum())

def standardize_df(df):
    # Standardize data
    df_std = (df - df.mean()) / df.std()
    return df_std

def train_test_split(df, test_size=0.2):
    # Shuffle data
    df = df.sample(frac=1).reset_index(drop=True)

    # Split data
    train = df[:int(len(df)*(1-test_size))]
    test = df[int(len(df)*(1-test_size)):]

    return train, test

def get_mutual_information(xdf, y):
    """ Calculate the mutual information for each variable.
    This is close to seeing how much each variable can tell us about the class.
    """
    # Calculate mutual information
    mutual_info = mutual_info_classif(xdf, y,copy=True,discrete_features=False)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = xdf.columns
    mutual_info.sort_values(ascending=False, inplace=True)
    return mutual_info

    

if __name__ == "__main__":
    # Read data
    df = pd.read_csv("Acoustic-Features.csv")
    print_basic_summary(df)
    train_df, test_df = train_test_split(df, test_size=0.25)
    # Encode class
    mapping = {"relax":0, "happy":1, "sad":2, "angry":3}
    train_y = train_df.pop("Class").map(mapping)
    test_y = test_df.pop("Class").map(mapping)
    standardize_df(train_df)
    mutual_info = get_mutual_information(train_df, train_y)
    print(mutual_info)







