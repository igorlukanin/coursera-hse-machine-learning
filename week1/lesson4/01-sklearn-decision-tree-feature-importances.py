from __future__ import print_function
from sklearn.tree import DecisionTreeClassifier

import pandas, pprint

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

featuresAndResult = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].dropna()
featuresAndResult.rename(columns={'Sex': 'SexString'}, inplace=True)
featuresAndResult['Sex'] = featuresAndResult['SexString'].map(lambda x: 1 if x == 'male' else -1)

features = featuresAndResult[['Pclass', 'Fare', 'Age', 'Sex']]
result = featuresAndResult['Survived']

classifier = DecisionTreeClassifier(random_state=241)
classifier.fit(features, result)

featureNames = features.keys().values
featureImportances = classifier.feature_importances_

importantFeatures = sorted(zip(featureNames, featureImportances), key=lambda x: x[1], reverse=True)
twoMostImportantFeatures = map(lambda x: x[0], importantFeatures[:2])

print()
print('Feature importances:')
pprint.PrettyPrinter().pprint(importantFeatures)

print()
print('2 most important features:', twoMostImportantFeatures)

file = open('01-result.txt', 'w')
print(' '.join(twoMostImportantFeatures), file=file, sep='', end='')
file.close()
