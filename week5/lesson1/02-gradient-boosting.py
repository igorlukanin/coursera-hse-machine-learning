from __future__ import print_function
from pandas import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss


def gb_get_min_loss(clf, verbose=False):
    j = 0
    min_loss_test = 1

    print()
    for i, quality_train, quality_test in zip(
            range(1, 250 + 1),
            clf.staged_predict_proba(X_train),
            clf.staged_predict_proba(X_test)
    ):
        loss_train = log_loss(y_train, quality_train)
        loss_test = log_loss(y_test, quality_test)

        if min_loss_test > loss_test:
            min_loss_test = loss_test
            j = i

            if (verbose):
                print(
                    'Iteration:', i, '  ',
                    'Train:', '{0:.3f}'.format(loss_train), '  ',
                    'Test:', '{0:.3f}'.format(loss_test), '  ',
                    '-' if min_loss_test == loss_test else '+'
                )

    return min_loss_test, j


data = read_csv('gbm-data.csv').values

y = data[:,0]
X = data[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)


clf = GradientBoostingClassifier(learning_rate=0.2, n_estimators=250, random_state=241, verbose=False)
clf.fit(X_train, y_train)
gb_min_loss, gb_j = gb_get_min_loss(clf)

print()
print('Iteration:', gb_j)
print('Gradient boosting min loss:', gb_min_loss)


clf = RandomForestClassifier(n_estimators=gb_j, random_state=241)
clf.fit(X_train, y_train)
rf_min_loss = log_loss(y_test, clf.predict_proba(X_test))

print('Random forest min loss:', rf_min_loss)


file = open('02-result-01.txt', 'w')
print('overfitting', file=file, sep='', end='')
file.close()

file = open('02-result-02.txt', 'w')
print('{0:.2f} {1}'.format(gb_min_loss, gb_j), file=file, sep='', end='')
file.close()

file = open('02-result-03.txt', 'w')
print('{0:.2f}'.format(rf_min_loss), file=file, sep='', end='')
file.close()