import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn import svm
from warnings import filterwarnings
import numpy as np
import matplotlib.pyplot as plt
filterwarnings("ignore") 

df = pd.read_csv('data/train.csv')

scaler = StandardScaler()
encoder = LabelEncoder()
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
scaler.fit(df[['Age']])
encoder.fit(df['Sex'])
imputer.fit(df[['Age']])
df['Sex'] = encoder.transform(df['Sex'])
df['Age'] = scaler.transform(df[['Age']])
df['Age'] = imputer.transform(df[['Age']])

X = df[['Sex', 'Age', 'Pclass']]

model = svm.SVC(kernel='linear')
model.fit(X.values, df['Survived'].values)

# sex = input('Enter sex(male/female): ')
# age = int(input('Enter age: '))
# pclass = int(input('Enter pclass: '))

# sex = encoder.transform([sex])[0]
# age = scaler.transform([[age]])[0][0]
# print(model.predict([[sex, age, pclass]]))

plt.scatter(df['Age'], df['Survived'])
plt.show()