from __future__ import print_function
from numpy import mean
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import scale
from pandas import read_csv


def calculate_score(n_neighbors, features, result, cv):
    classifier = KNeighborsClassifier(n_neighbors)
    score = cross_val_score(classifier, features, result, 'accuracy', cv)
    return mean(score)


data = read_csv('wine.csv')

result = data[data.columns[0]]
features = data[data.columns[1:]]
scaledFeatures = scale(features)

fold = KFold(result.size, n_folds=5, shuffle=True, random_state=42)


k_accuracy = {}

for i in range(1, 50 + 1):
    k_accuracy[i] = calculate_score(i, features, result, fold)

k_with_best_accuracy = max(k_accuracy, key=k_accuracy.get)
best_accuracy = k_accuracy[k_with_best_accuracy]

print()
print('Best accuracy:', best_accuracy)
print('k =', k_with_best_accuracy)


k_accuracy2 = {}

for i in range(1, 50 + 1):
    k_accuracy2[i] = calculate_score(i, scaledFeatures, result, fold)

k_with_best_accuracy2 = max(k_accuracy2, key=k_accuracy2.get)
best_accuracy2 = k_accuracy2[k_with_best_accuracy2]

print()
print('Best accuracy (scaled features):', best_accuracy2)
print('k =', k_with_best_accuracy2)


file = open('01-result-01.txt', 'w')
print(k_with_best_accuracy, file=file, sep='', end='')
file.close()

file = open('01-result-02.txt', 'w')
print('{0:.2f}'.format(best_accuracy), file=file, sep='', end='')
file.close()

file = open('01-result-03.txt', 'w')
print(k_with_best_accuracy2, file=file, sep='', end='')
file.close()

file = open('01-result-04.txt', 'w')
print('{0:.2f}'.format(best_accuracy2), file=file, sep='', end='')
file.close()
