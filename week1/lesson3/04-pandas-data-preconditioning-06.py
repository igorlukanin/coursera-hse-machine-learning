from __future__ import print_function

import pandas, re


def extract_female_first_name(name):
    match = re.match('([^,]+), ([^.]+)\. ([^()]+)?(\(([^")]+)\))? ?(\("([^)]+)"\))?', name)

    full_first_name = None

    if match:
        if match.group(5): # Married, look at maiden name
            full_first_name = match.group(5)
        else: # Not married, look at common name
            full_first_name = match.group(3)

    # First name may contain multiple words, take the first
    first_name = full_first_name.split()[0]

    return first_name


data = pandas.read_csv('titanic.csv', index_col='PassengerId')

females = data[data['Sex'] == 'female']
femaleNames = females['Name']
femaleFirstNames = femaleNames.map(extract_female_first_name)
femaleFirstNamesCounts = femaleFirstNames.value_counts()
mostFrequentFemaleFirstName = femaleFirstNamesCounts.head(1).keys()[0]

print()
print('Most frequent female first name:', mostFrequentFemaleFirstName)

file = open('04-result-06.txt', 'w')
print(mostFrequentFemaleFirstName, file=file, sep='', end='')
file.close()
