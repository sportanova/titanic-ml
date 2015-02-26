import pandas as pd
import numpy as np
import pylab as P
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier 

class Titanic:
  def __init__ (self, estimators):
    self.forest = RandomForestClassifier(n_estimators = estimators)

  def clean_data(self, file):
    df = pd.read_csv(file, header=0)

    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    median_ages = np.zeros((2,3))
    median_ages

    for i in range(0, 2):
      for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

    df['AgeFill'] = df['Age']

    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df.loc[df['Fare'].isnull(), 'Fare'] = 0

    df['Age*Class'] = df.AgeFill * df.Pclass

    self.passengerIds = df['PassengerId']

    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age', 'PassengerId'], axis=1)
    return df

  def learn(self, df_file):
    train, test = train_test_split(df_file, test_size = 0.2)
    self.forest = self.forest.fit(train[0::, 1::], train[:,0])
    # output = self.forest.predict(test[0::, 1::])

  def predict(self, cleaned_file):
    output = self.forest.predict(cleaned_file.values[0::])
    return output.astype(int)

  def createCSV(self, predictions, passengerIds):
    df = pd.DataFrame(passengerIds)
    df['Survived'] = predictions
    df.to_csv('titanic_results.csv', index = False)
    return None

titanic = Titanic(150)
df_file = titanic.clean_data('train.csv')
titanic.learn(df_file)

cleaned_test_data = titanic.clean_data('test.csv')
predictions = titanic.predict(cleaned_test_data)

titanic.createCSV(predictions, titanic.passengerIds)