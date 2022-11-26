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
    #print(df['Ticket'].unique())
    # Doesn't look predictive; will not use this column

    # Explore Cabin Column
    #print(df['Cabin'].unique())
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
    # Cabin letter T doesn't exist in test.csv; drop that dummy
    X_encoded.drop(['CabinLetter_T'], axis=1, inplace=True)
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
    print(f"How many females survived? {len(df[(df['Survived'] == 1) & (df['IsFemale'] == 1)])}")
    print(f"How many females died? {len(df[(df['Survived'] == 0) & (df['IsFemale'] == 1)])}")
    print(f"How many males survived? {len(df[(df['Survived'] == 1) & (df['IsFemale'] == 0)])}")
    print(f"How many males died? {len(df[(df['Survived'] == 0) & (df['IsFemale'] == 0)])}")
    # Cross validation
    """
    params = {
        'max_depth': [3,4,5],
        'learning_rate': [0.1, 0.05, 0.01], # also known as 'eta'
        'gamma': [0, 0.25, 1],
        'reg_lambda': [0, 1, 10],
        'scale_pos_weight': [1,3,5] # xgboost recommends sum(negatives) / sum(positives) 1.6
    }
    grid = GridSearchCV(
        estimator = clf_xgb,
        param_grid=params,
        scoring='roc_auc',
        verbose=True,
        n_jobs=10,
        cv=5
    )
    grid.fit(X_train, y_train)

    #print(grid.cv_results_)
    print("Best estimator:")
    print(grid.best_estimator_)
    print("Best score:")
    print(grid.best_score_)
    print("Best params:")
    print(grid.best_params_)
    

    # Best params:
    # {'gamma': 0.25, 'learning_rate': 0.05, 'max_depth': 4, 'reg_lambda': 10, 'scale_pos_weight': 1}
    # adjusting reg_lambda up
    # seeing if scale_pos_weight can be a float

    params = {
        'max_depth': [3,4,5],
        'learning_rate': [0.1, 0.05, 0.01], # also known as 'eta'
        'gamma': [0, 0.25, 1],
        'reg_lambda': [10, 20, 100],
        'scale_pos_weight': [1,1.5, 2] # xgboost recommends sum(negatives) / sum(positives) 1.6
    }
    grid = GridSearchCV(
        estimator = clf_xgb,
        param_grid=params,
        scoring='roc_auc',
        verbose=True,
        n_jobs=10,
        cv=5
    )
    grid.fit(X_train, y_train)

    #print(grid.cv_results_)
    print("Best estimator:")
    print(grid.best_estimator_)
    print("Best score:")
    print(grid.best_score_)
    print("Best params:")
    print(grid.best_params_)
    """

    params = {
        'max_depth': [5],
        'learning_rate': [0.05], # also known as 'eta'
        'gamma': [0.25],
        'reg_lambda': [20],
        'scale_pos_weight': [1] # xgboost recommends sum(negatives) / sum(positives) 1.6
    }
    grid = GridSearchCV(
        estimator = clf_xgb,
        param_grid=params,
        scoring='roc_auc',
        verbose=True,
        n_jobs=10,
        cv=5
    )
    grid.fit(X_train, y_train)

    #print(grid.cv_results_)
    print("Best estimator:")
    print(grid.best_estimator_)
    print("Best score:")
    print(grid.best_score_)
    print("Best params:")
    print(grid.best_params_)

    # Build optimal model
    clf_xgb = xgb.XGBClassifier(
        seed=42,
        objective='binary:logistic',
        gamma=0.25,
        learning_rate=0.05,
        max_depth=5,
        reg_lambda=20,
        scale_pos_weight=1,
        subsample=0.9,
        colsample_bytree=1
    )

    clf_xgb.fit(X_train,
        y_train,
        verbose=True,
        early_stopping_rounds=10,
        eval_metric='aucpr',
        eval_set=[(X_test, y_test)])

    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(clf_xgb,
        X_test,
        y_test,
        values_format='d',
        display_labels=["Did not survive", "Survived"])
    # TODO: am I sure about the labels?
    print("Saving confusion matrix...")
    plt.savefig('confusion_matrix_2.png')
    print("Confusion matrix saved.")

    print("Plotting feature importance...")
    xgb.plot_importance(clf_xgb)
    plt.savefig('feature_importance.png')

    # Load test csv
    df_test = pd.read_csv('data/test.csv')

    # Create submission csv
    # Change sex to IsFemale (bool) (less memory)
    replace_values = {'female': 1, 'male': 0}
    df_test['IsFemale'] = df_test['Sex'].copy().replace(replace_values).astype(bool)
    df_test['CabinLetter'] = [x[0] if not isinstance(x, float) else x for x in df_test['Cabin']]
    X_new = df_test.drop(['Name', 'Sex', 'Ticket', 'PassengerId', 'Cabin'], axis=1).copy()
    X_new_encoded = pd.get_dummies(X_new, columns=['Embarked', 'CabinLetter'])

    print("X_encoded columns:")
    print(X_encoded.dtypes)
    print("X_new_encoded columns:")
    print(X_new_encoded.dtypes)

    yhat = clf_xgb.predict(X_new_encoded)
    print(df_test['PassengerId'])
    print(len(df_test['PassengerId']))
    print(yhat)
    print(len(yhat))
    yhat_series = pd.Series(yhat, name='Survived')

    submission = df_test['PassengerId'].to_frame().join(yhat_series)[['PassengerId', 'Survived']]
    submission.to_csv('data/results.csv')

    full_result = pd.concat([X_new_encoded, submission], axis=1)
    full_result.to_csv('data/full_result.csv')


if __name__ == "__main__":
    main()