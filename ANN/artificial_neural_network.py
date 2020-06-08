
#Part 1: Data importing and preprocessing

import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values


#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#To avoid dummy variable trap
X = X[: , 1:]


#splitting the dataset into training set  and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)



#Featue Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#Part 2: Now lets make the ANN

import keras
from keras.models import Sequential
from keras.layers import Dense


#Initializing the ANN
classifier = Sequential()

#Adding the input layer and first hidden layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

#adding the second hidden layer
classifier.add(Dense(activation = 'relu', units=6, kernel_initializer="uniform"))

#adding the output layer
classifier.add(Dense(activation = 'sigmoid', units=1, kernel_initializer="uniform" ))

#compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics= ['accuracy'])

#Fitting the ANN to training set
classifier.fit(X_train, Y_train, batch_size=10 ,epochs=100)



#Part 3: Making the prediction and evaluating the model
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred>0.5)

#Making the confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
print(cm)

#finding accuracy
accuracy= ((cm[0][0]+cm[1][1])/2000)*100
print(accuracy)
