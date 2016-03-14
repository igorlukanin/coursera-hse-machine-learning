from __future__ import print_function
from datetime import datetime
from pandas import read_csv, DataFrame
from sklearn.cross_validation import KFold, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
from numpy import logspace, zeros, hstack

run_gradient_boosting = True
run_logistic_regression_for_all_features = True
run_logistic_regression_for_truncated_features = True


### Part 1. Gradient boosting

# 1. Loading features, removing match result features

match_result_features = [
    'duration',
    'tower_status_radiant',
    'tower_status_dire',
    'barracks_status_radiant',
    'barracks_status_dire'
]

def read_data(path):
    features = read_csv(path, index_col='match_id')
    features.drop('start_time', axis=1, inplace=True) # Irrelevant feature because a match may start at any time

    return features

features = read_data('./features.csv')
features.drop(match_result_features, axis=1, inplace=True)

print('\nTrain data. {0} rows, {1} features'.format(features.shape[0], features.shape[1]))


# 2. Selecting features with NA values

feature_counts = features.count()
na_features = feature_counts[feature_counts != features.shape[0]].index.values

print('\nPartially missing features:')
for name in na_features: print(' - {0}'.format(name))


# 3. Filling values of those features with zeroes

for name in na_features: features[name].fillna(0, inplace=True)


# 4. Extracting the target value from the set of features

target = 'radiant_win'

y = features[target]
features.drop(target, axis=1, inplace=True)
X = features


# 5. Performing the gradient boosting over 5 folds with cross-validation using ROC-AUC scoring

folds = KFold(features.shape[0], n_folds=5, shuffle=True)

if run_gradient_boosting:
    tree_counts = [10, 20, 30]

    class clf_wrapper_gb(GradientBoostingClassifier):
        def predict(self, X):
            return GradientBoostingClassifier.predict_proba(self, X)

    print()
    for count in tree_counts:
        start_time = datetime.now()

        probas = cross_val_predict(clf_wrapper_gb(n_estimators=count), X, y, folds)
        score = roc_auc_score(y, probas[:,1])

        elapsed_time = datetime.now() - start_time

        print('{0} trees, {1} to fit, ROC-AUC score: {2:.2f}'.format(count, elapsed_time, score))


### Part 2. Logistic regression

# 1. Scaling features because the logistic regression is a linear algorithm.
# Performing the logistic regression over the folds with cross-validation using ROC-AUC scoring

X_scaled = scale(X)


class clf_wrapper_lr(LogisticRegression):
    def predict(self, X):
        return LogisticRegression.predict_proba(self, X)

def calculate_best_score(X, y, folds):
    best_score = -1
    best_c = 0
    best_time = 0

    print()
    for c in logspace(-3, 0, num=5):
        start_time = datetime.now()

        clf = clf_wrapper_lr(penalty='l2', C=c)
        probas = cross_val_predict(clf, X, y, folds)
        score = roc_auc_score(y, probas[:,1])

        elapsed_time = datetime.now() - start_time

        print('C={0:.5f}, {1} to fit, ROC-AUC score: {2:.4f}'.format(c, elapsed_time, score))

        if score > best_score:
            best_score = score
            best_c = c
            best_time = elapsed_time

    return best_score, best_c, best_time

if run_logistic_regression_for_all_features:
    score, c, time = calculate_best_score(X_scaled, y, folds)
    print('\nAll features. Best C={0:.5f}, {1} to fit, ROC-AUC score: {2:.4f}'.format(c, time, score))


# 2. Removing categorial features and re-running the regression

non_hero_features = [
    'lobby_type'
]

hero_features = [
    'r1_hero',
    'r2_hero',
    'r3_hero',
    'r4_hero',
    'r5_hero',
    'd1_hero',
    'd2_hero',
    'd3_hero',
    'd4_hero',
    'd5_hero'
]

truncated_features_1 = features.drop(non_hero_features, axis=1)
truncated_features_2 = features.drop(non_hero_features + hero_features, axis=1)
X_scaled = scale(truncated_features_2)

if run_logistic_regression_for_truncated_features:
    score, c, time = calculate_best_score(X_scaled, y, folds)
    print('\nNo categorial features. Best C={0:.5f}, {1} to fit, ROC-AUC score: {2:.4f}'.format(c, time, score))


# 3. Counting unique heroes

def get_unique_heroes(features):
    unique_heroes = set()

    for name in hero_features:
        unique_heroes.update(features[name].value_counts().index.values)

    return list(unique_heroes)

unique_heroes = get_unique_heroes(truncated_features_1)

print('\n{0} unique heroes'.format(len(unique_heroes)))


# 4. Adding new features for heroes using the bag-of-words approach

def get_new_hero_features(features, unique_heroes):
    X_new_hero_features = zeros((features.shape[0], len(unique_heroes)))

    for i, match_id in enumerate(features.index):
        for p in xrange(5):
            r_hero = features.ix[match_id, 'r%d_hero' % (p + 1)]
            X_new_hero_features[i, unique_heroes.index(r_hero)] = 1

            d_hero = features.ix[match_id, 'd%d_hero' % (p + 1)]
            X_new_hero_features[i, unique_heroes.index(d_hero)] = -1

    return X_new_hero_features

X_new_hero_features = get_new_hero_features(truncated_features_1, unique_heroes)


# 5. Re-running the regression with new features

X_scaled = hstack((X_scaled, X_new_hero_features))

score, c, time = calculate_best_score(X_scaled, y, folds)
print('\nNew features for heroes. Best C={0:.5f}, {1} to fit, ROC-AUC score: {2:.4f}'.format(c, time, score))


# 6. Making predictions with the best algorithm

features_test = read_data('./features_test.csv')

print('\nTest data. {0} rows, {1} features'.format(features_test.shape[0], features_test.shape[1]))

feature_test_counts = features_test.count()
na_features_test = feature_test_counts[feature_test_counts != features_test.shape[0]].index.values
for name in na_features_test: features_test[name].fillna(0, inplace=True)

unique_heroes_test = get_unique_heroes(features_test)
X_new_hero_features_test = get_new_hero_features(features_test, unique_heroes_test)
truncated_features_test = features_test.drop(non_hero_features + hero_features, axis=1)
X_test_scaled = hstack((scale(truncated_features_test), X_new_hero_features_test))

best_clf = LogisticRegression(penalty='l2', C=c)
best_clf.fit(X_scaled, y)

Y_pred = best_clf.predict(X_test_scaled)

df = DataFrame(index=truncated_features_test.index, columns=[target])
df[target] = best_clf.predict_proba(X_test_scaled)
df.to_csv('predictions.csv')