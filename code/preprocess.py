import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def display_all(df,max_rows=1000,max_cols=1000):
    '''
    Displays all data in dataframe with analysis of var stats
    '''
    with pd.option_context('display.max_rows',max_rows,'display.max_columns',max_cols):
        return df



if __name__ == '__main__':

    # LOAD DATA
    train_data = pd.read_csv('files/train.csv')
    test_data = pd.read_csv('files/test.csv')
    print('Train data: {}'.format(train_data.shape))
    print('Test data: {}'.format(test_data.shape))

    # INITIALISE DATAFRAME
    df = pd.concat([train_data, test_data], axis=0, sort=True)
    headers = df.columns.values.tolist()
    print('Headers:\n{}'.format(headers))
    print('Initial DataFrame:\n{}'.format(df))

    # PROCESS VARIABLES
    print('Two types of variables: \ni) Categorical\nii) Continuous')
    
    # cat vars
    print('Categorical variables to keep: i) Embarked, ii) Sex')

    print('i) Sex: Binary categorical therefore encode as single 1-0 col')
    df['Sex'] = df['Sex'].astype('category') # conv to category dtype
    df['Sex'] = df['Sex'].cat.codes # conv to 1s & 0s
    print(df['Sex'])

    print('ii) Embarked: Multivariate cat therefore 1-hot encode')
    embarked_encoded = pd.get_dummies(df['Embarked'], prefix='Embarked') # one hot encode
    df = pd.concat([df, embarked_encoded], axis=1) # append encoded embarked
    del df['Embarked'] # remove original embarked
    print(df)
    
    df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
    print('Removed remaining categorical variables we don\'t want')

    # cont vars
    print('Continuous variables to keep: i) Age ii) Fare iii) Parch iv) SibSp')
    print('For NN, cont vars must be standardised w/ centering and scaling')
    cont_vars = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp']
    scaler = StandardScaler()
    for var in cont_vars:
        df[var] = df[var].astype('float64') # conv data to float
        df[var] = scaler.fit_transform(df[var].values.reshape(-1,1)) # center data and transform
    print(df)

    print('PROBLEM: Some Age params are empty (i.e. age unknown). NaN values crash NN training.')
    print('For rough fix, just take median of other ages and set NaN to this val')
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)

    # DISPLAY PREPROCESSED DATA
    print('Processed Data:\n{}'.format(df))
    print('Analysis:\n{}'.format(display_all(df.describe(include='all').T)))
    print('Headers:\n{}'.format(df.columns.values.tolist()))
    print('Shape: {}'.format(df.shape))    
    
    # SAVE PROCESSED DATA
    df.to_csv('files/processed_data.csv', index=False)








