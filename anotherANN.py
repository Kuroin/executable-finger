import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

#Encoding Label
le=LabelEncoder()
#Divide dataset to subsets
kfold=KFold(n_splits=5)
#Scaling numbers
sc = StandardScaler()


def train(path):

    dataset = pd.read_csv(path,encoding="utf8",dtype="unicode")

    dataset = dataset.T

    #Shuffle dataset

    dataset = dataset.sample(frac = 1).reset_index(drop=True)

    X= dataset.iloc[:, 1:].astype("float64").to_numpy()
    y = le.fit_transform(dataset.iloc[:, 0])

    X=sc.fit_transform(X)

    num_features = X.shape[1]

    #Neural Network

    model = Sequential()

    model.add(Dense(32,input_dim=num_features))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']    
    )

    #KFold Cross Validation
    accs =list()
    losses = list()
    for f,(train_idx,test_idx) in enumerate(kfold.split(X,y)):
        X_train,X_test = X[train_idx],X[test_idx]
        y_train,y_test = y[train_idx],y[test_idx]
        model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=32,
            validation_split=0.2
        )
        
        loss,accuracy = model.evaluate(X_test,y_test)
        accs.append(accuracy)
        losses.append(loss)


    print("Mean Acc: ", sum(accs)/len(accs))
    print("Mean Loss: ", sum(losses)/len(losses))

    y_pred = model.predict(X_test).round()
    class_report = classification_report(y_test, y_pred)
    cm= confusion_matrix(y_test, y_pred)

    sensivity = (cm[0][0])/sum(cm[0])+sum(cm[1])
    specificity = (cm[1][1])/sum(cm[0])+sum(cm[1])

    print("ANN Sensivity: ",sensivity)
    print("ANN Specificity: ",specificity)


if __name__ == "__main__":
    while True:
        path = input("Enter path to dataset (Enter 'q' to quit): ")
        if path == 'q':
            break
        else:
            train(path)
