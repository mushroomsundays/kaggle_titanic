import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

def main():
    df = pd.read_csv('data/train.csv')
    print(df.head())
    print(df.shape)
    print(df.dtypes)
    """
    (891, 12)
    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    """

    # Explore target column
    print(df['Survived'].unique()) # 0,1; no nulls

    print(f"How many females survived? {len(df[(df['Survived'] == 1) & (df['IsFemale'] == 1)])}")
    print(f"How many females died? {len(df[(df['Survived'] == 0) & (df['IsFemale'] == 1)])}")
    print(f"How many males survived? {len(df[(df['Survived'] == 1) & (df['IsFemale'] == 0)])}")
    print(f"How many males died? {len(df[(df['Survived'] == 0) & (df['IsFemale'] == 0)])}")

    # y is the target column we will use to build models
    y = df['Survived'].astype(bool).copy()

    # Look for missing values
    df_nulls = df.isna().sum()
    print(df_nulls)
    """
    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    """
    # Age and Cabin have a large number of nulls

    # Age is continuous; fill nulls with median
    df['Age'] = df['Age'].fillna(df['Age'].dropna().median())

    # Explore Cabin Column
    print(df['Cabin'].unique())
    # Cabin: C85, B28, etc. There are also NaN
    # Pull letter out of Cabin and treat as categorical (get dummy)
    # This deals with missing values by turning into 0 in dummy column
    df['CabinLetter'] = [x[0] if not isinstance(x, float) else x for x in df['Cabin']]
    print(df['CabinLetter'].unique())
    print(df['CabinLetter'].value_counts())
    """
    C    59
    B    47
    D    33
    E    32
    A    15
    F    13
    G     4
    T     1
    """

    # Change sex to IsFemale (bool) (less memory)
    replace_values = {'female': 1, 'male': 0}
    df['IsFemale'] = df['Sex'].copy().replace(replace_values).astype(bool)

    # Explore Ticket column
    #print(df['Ticket'].unique())
    # Doesn't look predictive; will not use this column

    X = df.drop(['Survived', 'Name', 'Sex', 'Ticket', 'PassengerId', 'Cabin'], axis=1).copy()

    # Save cleaned data to csvs
    X.to_csv('data/X.csv', index=False)
    y.to_csv('data/y.csv', index=False)


if __name__ == "__main__":
    main()