from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import xgboost as xgb

lr = LogisticRegression()
dc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()
svc= SVC()
gnb = GaussianNB()
bnb= BernoulliNB()
xg_reg = xgb.XGBClassifier(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

le=LabelEncoder()
sc = StandardScaler()

data = pd.read_csv("otu.csv", dtype = "unicode")
dataset = data.T
# dataset = dataset.sample(frac = 1).reset_index(drop=True)

X=dataset.iloc[:, 1:].astype("float64").to_numpy()

# X=sc.fit_transform(X)
y = le.fit_transform(dataset.iloc[:, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

models = [lr,dc,rfc,knn,svc,gnb,bnb,xg_reg]
#lr  LogisticRegression
#dc  DecisionTreeClassifier
#rfc RandomForestClassifier
#knn KNeighborsClassifier
#svc SVC
#gnb GaussianNB
#bnb BernoulliNB
#xg_reg XGBoost

for model in models:
    score = cross_val_score(model, X_train,y_train,cv=5).mean()
    loss = cross_val_score(model, X_train,y_train,cv=5).std()
    
    model.fit(X_train, y_train)
    y_pred= model.predict(X_test)
    cm=confusion_matrix(y_test, y_pred)
    
    sensivity = (cm[0][0])/sum(cm[0])+sum(cm[1])
    specificity = (cm[1][1])/sum(cm[0])+sum(cm[1])
    
    print(f"{model} Accuracy :", score)
    print(f"{model} Loss :", loss)
    print("\n")    
    print(f"{model} Sensivity: ",sensivity)
    print(f"{model} Specificity: ",specificity)
    print("\n") 
    

