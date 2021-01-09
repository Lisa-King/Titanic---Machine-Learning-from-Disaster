#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', 50)
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
df_train['Embarked'].fillna('S', inplace=True)

#Convert categorical variable into dummy/indicator variables.
embarked_dummies = pd.get_dummies(df_train['Embarked'], prefix='Embarked')
df_train = pd.concat([df_train, embarked_dummies], axis=1)
df_train.drop('Embarked', axis=1, inplace=True)
print(df_train.columns)

#There are too many missing values in Cabin, instead of removing this variable, we use 'Missing' to replace it
df_train['Cabin'].fillna('Missing', inplace=True)

#Also, the cabin contains numbers, but we care more about the Cabin class, so we will remove numbers.
# mapping each Cabin value with the cabin letter
df_train['Cabin'] = df_train['Cabin'].map(lambda x: x[0])
df_train['Cabin'].unique()

#Let's first check if correlation still works for this project
corrmat = df_train.corr()
sns.heatmap(corrmat, square=True,annot=True)
plt.show()

#Let plot the survived data and other features
df_train['Died'] = 1 - df_train['Survived']
#Pclass
df_train.groupby('Pclass').sum()[['Survived', 'Died']].plot(kind='bar', stacked=True, color=['g', 'r'])
#Sex
df_train.groupby('Sex').sum()[['Survived', 'Died']].plot(kind='bar', stacked=True, color=['g', 'r'])
#Age variable
#A nice way to compare distributions is to use a violin plot
sns.violinplot(x='Sex', y='Age', hue='Survived', data=df_train, split=True,palette={0: "r", 1: "g"})

#SibSp
df_train.groupby('SibSp').sum()[['Survived', 'Died']].plot(kind='bar', stacked=True, color=['g', 'r'])
#Cannot tell if SibSp matters becasue when SibSp >=2 the volumn is too small. We then choose to plot percenage.
SibSp_df=df_train[['SibSp','Survived', 'Died']].groupby('SibSp').sum()
SibSp_df['Survive_R']=SibSp_df['Survived']/(SibSp_df['Survived']+SibSp_df['Died'])
SibSp_df['Died_R']=1-SibSp_df['Survive_R']
SibSp_df.groupby('SibSp').sum()[['Survive_R', 'Died_R']].plot(kind='bar', stacked=True, color=['g', 'r'])

#Similarly, we will do the same for Parch
df_train.groupby('Parch').sum()[['Survived', 'Died']].plot(kind='bar', stacked=True, color=['g', 'r'])
Parch_df=df_train[['Parch','Survived', 'Died']].groupby('Parch').sum()
Parch_df['Survive_R']=Parch_df['Survived']/(Parch_df['Survived']+Parch_df['Died'])
Parch_df['Died_R']=1-Parch_df['Survive_R']
Parch_df.groupby('Parch').sum()[['Survive_R', 'Died_R']].plot(kind='bar', stacked=True, color=['g', 'r'])

#Ticket fare
plt.hist([df_train[df_train['Survived'] == 1]['Fare'], df_train[df_train['Survived'] == 0]['Fare']], stacked=True, color = ['g','r'], bins = 50, label = ['Survived','Died'])
plt.legend()

#Before we move futher, let's chech the correlation between input features
df_train.groupby('Pclass').mean()['Fare'].plot(kind='bar')
df_train.groupby('Pclass').mean()['Age'].plot(kind='bar')

#Ok, now we have got a general idea about our datasets. It is time to do feature engineering and selection
# reading test data
df_test = pd.read_csv('test.csv')

# extracting and then removing the targets from the training data
targets = df_train['Survived']
df_train.drop(['Survived'], 1, inplace=True)

# merging train data and test data for future feature engineering
# we'll also remove the PassengerID since this is not an informative feature
combined = df_train.append(df_test)
combined.reset_index(inplace=True)
combined.drop(['index', 'PassengerId'], inplace=True, axis=1)

#Check the combined dataset shape
print(combined.shape)

#check the first five rows in combined dataset
combined.head()

#get unique titles from our combined datasets. strip function is to remove the extra spaces.
titles = set()
for name in combined['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
print(titles)

#Now let's map the title can bin them
#Captain, Colonel, Major, Doctor, Reverend can be binned to officer
#Jonkheer, Don, Dona, Sir, the Countess, Lady can be binned to Royalty
#Madame, Ms, Mrs can be binned to Mrs
#Mademoiselle, , Miss can be binned to Miss
#Mr
#Master: male children: Young boys were formerly addressed as "Master [first name]."

Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

#Generate a new Title column
combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
combined['Title'] = combined['Title'].map(Title_Dictionary)

#check if there is any missing Title
combined[combined['Title'].isnull()]

#check missing age data in the training and test dataset
print(combined.iloc[:891]['Age'].isnull().sum())
print(combined.iloc[891:]['Age'].isnull().sum())

#let's get the median age based on people's gender, Pclass and Title
grouped_train = combined.iloc[:891].groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
print(grouped_median_train)

#Now we just need to map these medium ages to the missing parts. [0] is to convert list to number.
def fill_age(row):
    condition = (
        (grouped_median_train['Sex'] == row['Sex']) &
        (grouped_median_train['Title'] == row['Title']) &
        (grouped_median_train['Pclass'] == row['Pclass'])
    )
    return grouped_median_train[condition]['Age'].values[0]

combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
print(combined['Age'].isnull().sum())

# Name can be dropped now
combined.drop('Name', axis=1, inplace=True)

# encoding in dummy variable
titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
combined = pd.concat([combined, titles_dummies], axis=1)

# removing the title variable
combined.drop('Title', axis=1, inplace=True)

#Fill out the missing fare data
combined['Fare'].fillna(combined['Fare'].mean(), inplace=True)

# two missing embarked values - filling them with the most frequent one in the train set
combined['Embarked'].fillna('S', inplace=True)

# encoding in dummy variable
embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
combined = pd.concat([combined, embarked_dummies], axis=1)
combined.drop('Embarked', axis=1, inplace=True)

#Now let's test if there are different Cabin categories in the test dataset
#Why? If there are some Cabin categories in the test dataset but not in the training dataset, we have to replace them.
train_cabin, test_cabin = set(), set()

for c in combined.iloc[:891]['Cabin']:
    try:
        train_cabin.add(c[0])
    except:
        train_cabin.add('M')

for c in combined.iloc[891:]['Cabin']:
    try:
        test_cabin.add(c[0])
    except:
        test_cabin.add('M')
print(train_cabin)
print(test_cabin)

#Now let's fill out the missing values for Cabin
combined['Cabin'].fillna('M', inplace=True)
combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

# dummy encoding ...
cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
combined = pd.concat([combined, cabin_dummies], axis=1)
combined.drop('Cabin', axis=1, inplace=True)

#For now, check missing data
combined.isnull().sum()

# encoding into 3 categories:
pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

# adding dummy variable
combined = pd.concat([combined, pclass_dummies],axis=1)

# removing "Pclass"
combined.drop('Pclass',axis=1,inplace=True)

# mapping gender to numerical one
combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})

#Previously we have explored the SibSp and Parch, now we will merge these two together
# introducing a new feature : the size of families (including the passenger)
combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

# introducing other features based on the family size
combined['Single'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

#a function that extracts each prefix of the ticket, returns 'NONE' if no prefix (i.e the ticket is a digit)
def cleanTicket(ticket):
    ticket = ticket.replace('.','')
    ticket = ticket.replace('/','')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = [x for x in ticket if not x.isdigit()]
    if len(ticket) > 0:
        return ticket[0]
    else:
        return 'NONE'

#Get Ticket info
combined['Ticket'] = combined['Ticket'].map(cleanTicket)

# Extracting dummy variables from tickets:
tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
combined = pd.concat([combined, tickets_dummies], axis=1)
combined.drop('Ticket', inplace=True, axis=1)

#Check current dataset
print(combined.shape)

#Prepare the training dataset
df_im_input=combined.iloc[:891]
df_im_output=targets

#Now let's get the importance of each feature
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(df_im_input, df_im_output)

features = pd.DataFrame()
features['feature'] = df_im_input.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)

#plot it
features.plot(kind='barh', figsize=(25, 25))
plt.yticks()
plt.show()

#select top 10 important features
top_10_feature=features.nlargest(10, 'importance')

#choose model final input features
df_input_final=df_im_input[top_10_feature['feature']]

#build Logistic Regression Model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(df_input_final,targets)

#Let’s first test our training dataset prediction accuracy
#get predictions based on training input
preds=logreg.predict(df_input_final)
preds_probabilities = logreg.predict_proba(df_input_final)

#preds_probabilities has two numbers for each row of features: [probability of false, probability of true]
preds_probabilities.shape

#just need one as they can be calcualted using 1- other
pred_probs = preds_probabilities[:, 1]

from sklearn.metrics import roc_curve, auc
#roc_curve() returns a list of false positive rates (FPR) and true positives rates (TPR) for different configurations of the classifier used to plot the ROC.
[fpr, tpr, thr] = roc_curve(targets, pred_probs)

#plot ROC curve
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=14)
plt.ylabel('True Positive Rate (TPR)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

#check model accuracy on training dataset
from sklearn.metrics import confusion_matrix, accuracy_score
print("accuracy: %2.3f" % accuracy_score(targets, preds))
print("AUC: %2.3f" % auc(fpr, tpr))

#confusion matrix can give us the number of true positives, false positives, true negatives, and false negatives.
conf_m=confusion_matrix(targets, preds)

#get the test input and predictions
df_test_input_final=combined.iloc[891:][top_10_feature['feature']]
df_test_preds=logreg.predict(df_test_input_final)

#output the results to a csv file
submit = pd.DataFrame()
test = pd.read_csv('test.csv')
submit['PassengerId'] = test['PassengerId']
submit['Survived'] = df_test_preds
submit.to_csv('Titanic_LR_20200624.csv', index=False)


#15 features
#select top 15 important features
top_15_feature=features.nlargest(15, 'importance')
df_input_final=df_im_input[top_15_feature['feature']]

#build Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(df_input_final,targets)

#get predictions based on training input
preds=logreg.predict(df_input_final)
preds_probabilities = logreg.predict_proba(df_input_final)
pred_probs = preds_probabilities[:, 1]

[fpr, tpr, thr] = roc_curve(targets, pred_probs)

#plot ROC curve
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=14)
plt.ylabel('True Positive Rate (TPR)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

#check model accuracy on training dataset
from sklearn.metrics import confusion_matrix, accuracy_score
print("accuracy: %2.3f" % accuracy_score(targets, preds))
print("AUC: %2.3f" % auc(fpr, tpr))

conf_m=confusion_matrix(targets, preds)

#get the test input and predictions
df_test_input_final=combined.iloc[891:][top_15_feature['feature']]
df_test_preds=logreg.predict(df_test_input_final)

#output the results to a csv file
submit = pd.DataFrame()
test = pd.read_csv('test.csv')
submit['PassengerId'] = test['PassengerId']
submit['Survived'] = df_test_preds
submit.to_csv('Titanic_LR_15_20200624.csv', index=False)





