#Import packages
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)

#Load the training data
df_train=pd.read_csv('train.csv')

#Check data shape
print(df_train.shape)

#check first five rows
df_train.head()

#Check missing data
df_train.isnull().sum()

#Fill in the missing age with median age number
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())

#Only 2 Embarked is missing. Let's remove these two rows
df_train['Embarked'].unique()
df_train=df_train[df_train['Embarked'].notna()]

#Another way is to replace these two with the most frequent one 'S'
df_train['Embarked'].value_counts().plot(kind='bar')
df_train['Embarked'].fillna('S',inplace=True)
plt.show()

#Convert categorical variable into dummy/indicator variables.
embarked_dummies = pd.get_dummies(df_train['Embarked'], prefix='Embarked')
df_train = pd.concat([df_train, embarked_dummies], axis=1)
df_train.drop('Embarked', axis=1, inplace=True)

#There are too many missing values in Cabin, instead of removing this variable, we use 'Missing' to replace it
df_train['Cabin'].fillna('Missing', inplace=True)

#Also, the cabin contains numbers, but we care more about the Cabin class, so we will remove numbers.
# mapping each Cabin value with the cabin letter
df_train['Cabin'] = df_train['Cabin'].map(lambda x: x[0])
df_train['Cabin'].unique()
