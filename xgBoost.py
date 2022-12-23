import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.metrics import accuracy_score

kfold=KFold(n_splits=5)
data = pd.read_csv("otu.csv", dtype = "unicode")
dataset = data.T

dataset = dataset.sample(frac = 1).reset_index(drop=True)

le=LabelEncoder()
sc = StandardScaler()


X=dataset.iloc[:, 1:].astype("float64").to_numpy()

#Etiketlerimi encodeladım
y = le.fit_transform(dataset.iloc[:, 0])

#Sayıları 0 ile 1 arasına sığdırdım
X=sc.fit_transform(X)

data_dmatrix = xgb.DMatrix(data=X, label=y)


xg_reg = xgb.XGBClassifier(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

accs =list()
losses = list()
for f,(train_idx,test_idx) in enumerate(kfold.split(X,y)):
    print("Fold ",f+1)
    X_train,X_test = X[train_idx],X[test_idx]
    y_train,y_test = y[train_idx],y[test_idx]

    print(f"Fold {f+1} training")
    xg_reg.fit(X_train,y_train)
    y_pred=xg_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Fold Acc: ",accuracy)
    accs.append(accuracy)



print("XGBoost Mean Acc: ", sum(accs)/len(accs))


preds = xg_reg.predict(X_test).round()

score = accuracy_score(y_test, preds)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, preds)

sensivity = (cm[0][0])/sum(cm[0])+sum(cm[1])
specificity = (cm[1][1])/sum(cm[0])+sum(cm[1])

print("XGBoost Sensivity: ",sensivity)
print("XGBoost Specificity: ",specificity)
