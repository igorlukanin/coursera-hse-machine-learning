from __future__ import print_function
from pandas import read_csv
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler


data_train = read_csv('perceptron-train.csv', header=None)
data_test = read_csv('perceptron-test.csv', header=None)

y_train = data_train[data_train.columns[0]]
y_test = data_test[data_test.columns[0]]

X_train = data_train[data_train.columns[1:]]
X_test = data_test[data_test.columns[1:]]


perceptron = Perceptron(random_state=241)
perceptron.fit(X_train, y_train)

accuracy = perceptron.score(X_test, y_test)

print()
print('Accuracy:', accuracy)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

perceptron2 = Perceptron(random_state=241)
perceptron2.fit(X_train_scaled, y_train)

accuracy2 = perceptron2.score(X_test_scaled, y_test)
accuracy_delta = accuracy2 - accuracy

print()
print('Accuracy (scaled features):', accuracy2)
print('Accuracy delta:', accuracy_delta)


file = open('01-result.txt', 'w')
print('{0:.3f}'.format(accuracy_delta), file=file, sep='', end='')
file.close()