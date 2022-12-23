import pandas as pd
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report
gnb = GaussianNB()


le=LabelEncoder()
sc = StandardScaler()

data = pd.read_csv("otu.csv", dtype = "unicode")
dataset = data.T
dataset = dataset.sample(frac = 1).reset_index(drop=True)

X=dataset.iloc[:, 1:].astype("float64").to_numpy()

# X=sc.fit_transform(X)
y = le.fit_transform(dataset.iloc[:, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

gnb.fit(X_train, y_train)

y_pred=gnb.predict(X_test)

score = accuracy_score(y_test, y_pred)

class_report = classification_report(y_test, y_pred)
