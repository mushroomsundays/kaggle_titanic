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
    # Fill Embarked with the most likely value (S)
    df['Embarked'].fillna('S', inplace=True)

    # Age is continuous; fill nulls with median
    df['Age'] = df['Age'].fillna(df['Age'].dropna().median())
    # After plotting age vs. survival, we see that children under 16 survived far more often
    # Create a column IsChild
    df['IsChild'] = np.where(df['Age'] <= 16, 1, 0).astype(bool)

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
    # Fill null cabin with C, the most commonly occurring
    df['CabinLetter'] = df['CabinLetter'].fillna('C')

    # SibSp and Parch both relate to traveling with family, so combine them
    # to account for multicollinearity and for simplicity
    df['IsAlone']=np.where((df["SibSp"]+df["Parch"])>0, 0, 1)

    # Change sex to IsFemale (bool) (less memory)
    replace_values = {'female': 1, 'male': 0}
    df['IsFemale'] = df['Sex'].copy().replace(replace_values).astype(bool)

    print(f"How many females survived? {len(df[(df['Survived'] == 1) & (df['IsFemale'] == 1)])}")
    print(f"How many females died? {len(df[(df['Survived'] == 0) & (df['IsFemale'] == 1)])}")
    print(f"How many males survived? {len(df[(df['Survived'] == 1) & (df['IsFemale'] == 0)])}")
    print(f"How many males died? {len(df[(df['Survived'] == 0) & (df['IsFemale'] == 0)])}")

    # Drop unused columns

    X = df.drop(['Survived', 'Name', 'Sex', 'Ticket', 'PassengerId', 'Cabin', 'SibSp', 'Parch'], axis=1).copy()

    # One hot encoding
    X_encoded = pd.get_dummies(X, columns=['Embarked', 'CabinLetter', 'Pclass'])
    # Cabin letter T doesn't exist in test.csv; drop that dummy
    X_encoded.drop(['CabinLetter_T'], axis=1, inplace=True)
    # We can always drop one categorical column; if the others are 0 then we know the person
    # falls into the last category
    X_encoded.drop(['Embarked_Q', 'CabinLetter_G', 'Pclass_3'], axis=1, inplace=True)
    print(X_encoded.dtypes)
    print(y.dtypes)
    # check for nulls in X_encoded
    print(X_encoded.isna().any())

    # Save cleaned data to csvs
    X_encoded.to_csv('data/X.csv', index=False)
    y.to_csv('data/y.csv', index=False)

    ########################################
    # Ideas from Kaggle Titanic articles on Medium etc.
    ########################################
    # Extract title from passenger name (Capt, Lady, Mr, Sir, etc)
    # For age, instead of filling with median, fill with median grouped by sex, class, and title

if __name__ == "__main__":
    main()