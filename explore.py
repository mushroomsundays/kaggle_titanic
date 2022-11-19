import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score # split data
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer # for scoring during model selection
from sklearn.model_selection import GridSearchCV, RepeatedKFold # cross validation
from sklearn.metrics import confusion_matrix # creates a confusion matrix
from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix
from sklearn.metrics import mean_squared_error as MSE

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

    # Explore target
    print(df['Survived'].unique()) # 0,1
    y = df['Survived'].astype(bool).copy()

    # Change sex to IsFemale (bool) (less memory)
    replace_values = {'female': 1, 'male': 0}
    df['IsFemale'] = df['Sex'].copy().replace(replace_values).astype(bool)

    # Explore Ticket column
    print(df['Ticket'].unique())
    # Doesn't look predictive; will not use this column

    # Explore Cabin Column
    print(df['Cabin'].unique())
    # Cabin: C85, B28, etc. There are also NaN
    # 687 / 891 are NaN
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
    X = df.drop(['Survived', 'Name', 'Sex', 'Ticket', 'PassengerId', 'Cabin'], axis=1).copy()
    X_encoded = pd.get_dummies(X, columns=['Embarked', 'CabinLetter'])
    print(X_encoded.dtypes)
    print(y.dtypes)
    # check for nulls in X_encoded
    # Age has nulls still
    X_encoded = X_encoded.replace({np.nan: 0})
    print(X_encoded.isna().any())

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)

    # Build the model, a forest of extreme gradient boosted trees
    clf_xgb = xgb.XGBClassifier(objective='binary:logistic', missing=1, seed=42)
    print("Fitting model...")
    # TODO: early_stopping_rounds depricated
    # plot_confusion_matrix depricated
    clf_xgb.fit(X_train,
        y_train,
        verbose=True,
        early_stopping_rounds=10,
        eval_metric='aucpr',
        eval_set=[(X_test, y_test)])
    """
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(clf_xgb,
        X_test,
        y_test,
        values_format='d',
        display_labels=["Did not survive", "Survived"])
    # TODO: am I sure about the labels?
    print("Saving confusion matrix...")
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved.")
    """
    print(f"How many survived? {len(df[df['Survived'] == 1])}")
    print(f"How many died? {len(df[df['Survived'] == 0])}")
    # Cross validation
    params = {
        'max_depth': [3,4,5],
        'learning_rate': [0.1, 0.05, 0.01], # also known as 'eta'
        'gamma': [0, 0.25, 1],
        'reg_lambda': [0, 1, 10],
        'scale_pos_weight': [1,3,5] # xgboost recommends sum(negatives) / sum(positives) 1.6
    }
    optimal_params = GridSearchCV(
        estimator = clf_xgb,
        param_grid=params,
        scoring='roc_auc',
        verbose=True,
        njobs=10,
        cv=5
    )

    # name, ticket, cabin are object
    # I think drop name
    # Ticket: looks pretty nonsense / non-predictive
    # Cabin is like 'C123' and has NaNs. WHat to do?
    #print(f"Cabin NaN count: {len(df[df['Cabin'].isna()])}")
    # I think we can pull the letter out of Cabin and then dummy it, so nulls become irrelevant
    #plt.scatter(df['Survived'], df['Fare'])
    #plt.savefig('fare_scatter.png')
    #print(df['Ticket'].unique())
    #print(df['Cabin'].head(20))
    #print(f"df length: {len(df)}")



if __name__ == "__main__":
    main()