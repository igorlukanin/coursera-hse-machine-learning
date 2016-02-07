from __future__ import print_function
from numpy import mean, linspace
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston


def calculate_score(p, features, result, cv):
    regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric='minkowski')
    score = cross_val_score(regressor, features, result, 'mean_squared_error', cv)
    return mean(score)


data = load_boston()

result = data.target
features = scale(data.data)

fold = KFold(result.size, n_folds=5, shuffle=True, random_state=42)

p_accuracy = {}

for i in linspace(start=1, stop=10, num=200):
    p_accuracy[i] = calculate_score(i, features, result, fold)

p_with_best_accuracy = max(p_accuracy, key=p_accuracy.get)
best_accuracy = p_accuracy[p_with_best_accuracy]

print()
print('Best accuracy:', best_accuracy)
print('k =', p_with_best_accuracy)


file = open('02-result.txt', 'w')
print('{0:.1f}'.format(p_with_best_accuracy), file=file, sep='', end='')
file.close()