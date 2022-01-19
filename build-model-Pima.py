import pandas as pd
pima = pd.read_csv('Pima Indians Diabetes Database.csv')

df = pima.copy()
target = 'Outcome'


# Separating X and y
X = df.drop('Outcome', axis=1)
Y = df['Outcome']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('Pima_clf.pkl', 'wb'))
