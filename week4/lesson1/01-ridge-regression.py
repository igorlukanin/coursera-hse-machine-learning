from __future__ import print_function
from pandas import read_csv
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


tfid_v = TfidfVectorizer(min_df=5)
dict_v1 = DictVectorizer()
dict_v2 = DictVectorizer()


def process_text_column(data, column, tfid, test=False):
    data[column] = data[column].str.lower().replace('[^a-zA-Z0-9]', ' ', regex=True)
    return tfid.transform(data[column]) if test else tfid.fit_transform(data[column])

def process_non_text_column(data, column, dict, test=False):
    data[column].fillna('nan', inplace=True)
    values = data[[column]].to_dict('records')
    return dict.transform(values) if test else dict.fit_transform(values)

def data_to_X(data, test=False):
    words = process_text_column(data, 'FullDescription', tfid_v, test)
    location = process_non_text_column(data, 'LocationNormalized', dict_v1, test)
    time = process_non_text_column(data, 'ContractTime', dict_v2, test)
    return hstack([location, time, words])


data = read_csv('salary-train.csv')
X = data_to_X(data)

print()
print('Train data size: ', data.shape)
print('Train features:  ', X.shape)


clf = Ridge(alpha=1)
clf.fit(X, data['SalaryNormalized'])


data_test = read_csv('salary-test-mini.csv')
X_test = data_to_X(data_test, test=True)

print()
print('Test data size:  ', data_test.shape)
print('Test features:   ', X_test.shape)

y_test = clf.predict(X_test)

print()
print('Salaries:        ', y_test)


file = open('01-result.txt', 'w')
print('{0:.2f} {1:.2f}'.format(y_test[0], y_test[1]), file=file, sep='', end='')
file.close()