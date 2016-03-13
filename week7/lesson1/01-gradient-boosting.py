from __future__ import print_function
from pandas import read_csv
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from numpy import mean
from datetime import datetime

# 1

match_result_features = [
    'duration',
    'tower_status_radiant',
    'tower_status_dire',
    'barracks_status_radiant',
    'barracks_status_dire'
]

features = read_csv('./features.csv', index_col='match_id')
features.drop(match_result_features, axis=1, inplace=True)

print('\n{0} rows, {1} features'.format(features.shape[0], features.shape[1]))


# 2

feature_counts = features.count()
na_features = feature_counts[feature_counts != features.shape[0]].index.values

print('\nPartially missing features:')
for name in na_features: print(' {0}'.format(name))


# 3

for name in na_features: features[name].fillna(0, inplace=True)

feature_counts = features.count()
na_features = feature_counts[feature_counts != features.shape[0]].index.values


# 4

target = 'radiant_win'

y = features[target]
X = features.drop(target, axis=1)

# 5

tree_counts = [10, 20, 30, 99]

folds = KFold(features.shape[0], n_folds=5, shuffle=True)

print()
for count in tree_counts:
    start_time = datetime.now()

    clf = GradientBoostingClassifier(n_estimators=count)
    score = mean(cross_val_score(clf, X, y, 'r2', folds))

    elapsed_time = datetime.now() - start_time

    print('{0} trees, {1} to fit, CV score: {2:.2f}'.format(count, elapsed_time, score))