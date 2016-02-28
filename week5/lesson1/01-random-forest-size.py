from __future__ import print_function
from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from numpy import mean


data = read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = data[data.columns[0:-1]].as_matrix()
y = data[data.columns[-1]].as_matrix()


fold = KFold(y.size, n_folds=5, shuffle=True, random_state=1)
j = 0

print()
for i in range(1, 50 + 1):
    clf = RandomForestRegressor(n_estimators=i, random_state=1)
    score = mean(cross_val_score(clf, X, y, 'r2', fold))

    print('Forest size:', str(i).rjust(2), '  Score:', '{0:.3f}'.format(score), '  > 0.52:', '+' if score > 0.52 else '-')

    if j == 0 and score > 0.52:
        j = i


file = open('01-result.txt', 'w')
print(j, file=file, sep='', end='')
file.close()