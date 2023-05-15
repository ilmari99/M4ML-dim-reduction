import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import seaborn as sns

""" This is just a file to analyze.
"""

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

def mutual_info_matrix(df):
    n = df.shape[1]
    mi_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    for i in range(n):
        for j in range(i+1, n):
            x = df.iloc[:, i].values.reshape(-1, 1)
            y = df.iloc[:, j].values
            if df.iloc[:, i].dtype == "object":
                mi = mutual_info_classif(x, y, discrete_features=True)[0]
            else:
                mi = mutual_info_regression(x, y)[0]
            mi_matrix.iloc[i, j] = mi
            mi_matrix.iloc[j, i] = mi
    return mi_matrix

def dim_red_CMI(X,y):
    """ Do forward feature selection, to choose the best variables
    using continuos.get_mi

    Select variables one by one, based on which set of variables
    has the highest mutual information with the class.
    """
    n = X.shape[1]
    selected_vars = []
    for i in range(n):
        mi = []
        for j in range(n):
            if j not in selected_vars:
                mi.append(continuous.get_mi(X[:, j], y))
            else:
                mi.append(0)
        # Select variable with the highest absolute mutual information
        selected_var = np.argmax(np.abs(mi))
        selected_vars.append(selected_var)
    return X[:, selected_vars],y



DF = pd.read_csv("Acoustic-Features.csv")
mapping = {"relax":0, "happy":1, "sad":2, "angry":3}
DF["Class"] = DF["Class"].map(mapping)
if __name__ == "__main__":
    # Read data
    df = pd.read_csv("Acoustic-Features.csv")
    # Map the "Class" column to integers
    mapping = {"relax":0, "happy":1, "sad":2, "angry":3}
    df["Class"] = df["Class"].map(mapping)
    DF = df.copy()
    print_basic_summary(df)
    train_df, test_df = train_test_split(df, test_size=0.25)
    # Encode class
    mapping = {"relax":0, "happy":1, "sad":2, "angry":3}
    train_y = train_df.pop("Class").map(mapping)
    test_y = test_df.pop("Class").map(mapping)
    train_df = standardize_df(train_df)
    #selected_vars = get_CMI(np.array(train_df), np.array(train_y))
    #print(selected_vars)
    mat = mutual_info_matrix(DF)
    # Add entropy to diagonal
    for i in range(mat.shape[0]):
        mat.iloc[i, i] = 0#mutual_info_regression(DF.iloc[:, i].values.reshape(-1, 1), DF.iloc[:, i].values)[0]
    # Save matrix to excel
    #mat.to_excel("mutual_info_matrix.xlsx")
    # Create an sns heatmap of the matrix
    #sns.heatmap(mat.astype(float), cmap="YlGnBu")
    #plt.show()
    y = DF.pop("Class")
    mutual_info = get_mutual_information(DF, y)
    top_n_cols = mutual_info.index[:5].values
    print("Top 5 columns: ")
    print(top_n_cols)
    smal_df = pd.concat([DF[top_n_cols], y], axis=1)
    inv_mapping = {0:"relax", 1:"happy", 2:"sad", 3:"angry"}
    def make_categ(q):
        # invert the mapping
        return inv_mapping[q]
    smal_df["Mood"] = smal_df["Class"].apply(make_categ)
    smal_df.pop("Class")
    #Create an sns pairplot of the top 5 features with most mutual information, and remove upper triangle
    sns.pairplot(smal_df, hue="Mood", corner=True)
    plt.show()
    #print("Mutual information: ")
    #print(mutual_info[:10])
    








