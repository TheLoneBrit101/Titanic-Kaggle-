# Titanic prediction
import os
os.chdir("C:\\Users\\Andre\\Documents\\Data Science\\Projects\\Titanic Machine Learning from Disaster")
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
#from sklearn import preprocessing

import statistics as stats

import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


# read data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
full = [train, test]
full = pd.concat(full)

# EDA
full.head()
full.describe()
full.info()

# Feature Engineering

# Grab title from passenger names
full['Title'] = full['Name'].str.replace("(.*, )|(\\..*)", "")
# Show title counts by sex
pd.crosstab(full.Sex,full.Title)
## title_list = full['Title'].unique().tolist()

# Reduce number of vars by replacing low count titles
full['Title'] = full['Title'].replace("Mme","Mrs")
full['Title'] = full['Title'].replace(["Mlle","Ms"],"Miss")
full['Title'] = full['Title'].replace("Master","Mr")
full['Title'] = full['Title'].replace(['Don', 'Dona', 'Major', 'Capt',
    'Jonkheer', 'Rev', 'Col', 'Dr', 'Lady', 'Sir', 'the Countess'],"Rare")

pd.crosstab(full.Sex,full.Title)

# Surname
full['Surname'] = full['Name'].apply(lambda x: x.split(",")[0])

# Deck
full['Deck'] = full['Cabin'].apply(lambda x: list(x)[0] if(pd.notnull(x)) else x)
full['Deck'] = full['Deck'].fillna("Unknown")

# Family size
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1

sns.catplot(x="FamilySize", kind="count", hue="Survived", data=full)

full['FamilySizeGroup'] = "U"
full['FamilySizeGroup'][full['FamilySize'] == 1] = "Single"
full['FamilySizeGroup'][(full['FamilySize'] > 1) &
    (full['FamilySize'] < 5)] = "Regular"
full['FamilySizeGroup'][full['FamilySize'] >= 5] = "Large"

mosaic(full, ['FamilySizeGroup', 'Survived'], title= "Family Size by Survival")
plt.show()

# Missing values
full.isnull().sum(axis = 0)

## Embarked NaN values
full['PassengerId'][full['Embarked'].isnull()]
### PassengerId 62 & 830

## Filter out the two obs and then boxplot of class, fare paid, and embarkment point
embarkFare = full[(full.PassengerId != 62) & (full.PassengerId != 830)]
sns.boxplot(x="Embarked",y="Fare",hue="Pclass",data=embarkFare)
plt.axhline(80, color="red", ls='--')
plt.show()

del embarkFare

### From graph, PassengerId 62 & 830 most likely embarked from C
full['Embarked'][(full['PassengerId'] == 62) | (full['PassengerId'] == 830)] = "C"

## Fare Nan value
full['PassengerId'][full['Fare'].isnull()]
### PassengerId 1044
dfFare = full[(full.PassengerId != 1044)]
dfCl3EmS = dfFare[(dfFare['Embarked'] == "S") & (dfFare['Pclass'] == 3)]
sns.kdeplot(dfCl3EmS.Fare, shade=True)
plt.axvline(stats.median(dfCl3EmS.Fare), color="red", ls='--')
plt.show()

## Replace NaN value for median of Fare for Class 3 and embarkment S
full['Fare'][(full['PassengerId'] == 1044)] = stats.median(dfCl3EmS.Fare)
del (dfFare, dfCl3EmS)

# Drop Ticket and Name columns
full = full.drop(['Name', 'Ticket', 'Cabin','Surname'], axis=1)
full.Sex = full.Sex.map({'male':0, 'female':1}).astype(int)

grid = sns.FacetGrid(full, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

full.Embarked = full.Embarked.map({"S":0, "C":1, "Q":2}).astype(int)
full.Deck = full.Deck.map({'Unknown':0, 'C':3, 'E':5, 'G':7, 'D':4, 'A':1, 'B':2, 'F':6, 'T':8}).astype(int)
full.Title = full.Title.map({'Mr':0, 'Mrs':1, 'Miss':2, 'Rare':3}).astype(int)
full.FamilySizeGroup = full.FamilySizeGroup.map({'Regular':1, 'Single':0, 'Large':2}).astype(int)

#############################################################
##################################################################
# Baseline
train, test = full[:891], full[891:]
test = test.drop('Survived',axis=1)

full_cl = train

Xcol = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Deck', 'Title', 'FamilySize', 'FamilySizeGroup']
Ycol = 'Survived'

X = full_cl.loc[:, Xcol]
Y = full_cl.loc[:, Ycol]

Xbase = X
Ybase = Y

rf = RandomForestClassifier(n_estimators=1000,
                           max_depth=None,
                           min_samples_split=10)

baseline_err = cross_val_score(rf, X, Y, cv=10, n_jobs=-1).mean()
print("[BASELINE] Estimated RF Test Error (n = {}, 10-fold CV): {}".format(len(X), baseline_err))

# DETERMINISTIC REGRESSION
def age_fill(df):
    df_cl = df
    df_reg = df.dropna()
    
    Xrcol = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Deck', 'Title', 'FamilySize', 'FamilySizeGroup']
    Yrcol = 'Age'
    X_reg = df_reg.loc[:, Xrcol]
    Y_reg = df_reg.loc[:, Yrcol]
    
    age_lm = LinearRegression()
    age_lm.fit(X_reg, Y_reg)
    abs_residuals = np.absolute(Y_reg - age_lm.predict(X_reg))
    
    nan_inds = df_cl.Age.isnull().nonzero()[0]
    
    for i in nan_inds:
        df_cl.at[i,'Age'] = age_lm.predict(df_cl.loc[i, Xrcol].values.reshape(1, -1))
    return df_cl

train = age_fill(train)
test = age_fill(test)

Xreg = train.loc[:, Xcol]
Yreg = train.loc[:, Ycol]
    
reg_err = cross_val_score(rf, Xreg, Yreg, cv=10, n_jobs=-1).mean()
print("[DETERMINISTIC REGRESSION] Estimated RF Test Error (n = {}, 10-fold CV): {}".format(len(Xreg), reg_err))

full = [train, test]
full = pd.concat(full)

grid = sns.FacetGrid(full, col='Sex', hue='Survived', palette="Set1")
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


full['Child'] = "U"
full['Child'][full['Age'] < 14] = "Child"
full['Child'][full['Age'] >= 14] = "Adult"
pd.crosstab(full.Child,full.Survived)

full['Mother'] = "Not Ma"
full['Mother'][(full['Sex'] == 1) & (full['Child'] == 'Adult') & (full['Title'] != 2) & (full['Parch'] > 0)] = "Mother"
pd.crosstab(full.Mother,full.Survived)

full.Child = full.Child.map({"Child":0, "Adult":1}).astype(int)
full.Mother = full.Mother.map({"Not Ma":0, "Mother":1}).astype(int)

train, test = full[:891], full[891:]
test = test.drop('Survived',axis=1)


# Descision Tree
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.copy()

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

# Submission
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
    
    
importances = decision_tree.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.barh(X_train.columns, importances[indices],
       color="r", align="center")
plt.yticks(X_train.columns)
plt.ylim([-1, X_train.shape[1]])
plt.show()

export_csv = submission.to_csv (r'submission.csv', index = None, header=True)
