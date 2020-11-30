##Rodar com python 3.6(keras-facenet-master)
import matplotlib
import numpy as np
import pandas as pd
from pasta.augment import inline
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


#import matplotlib.pytrol as plt
#%matplotlib inline


df = pd.read_csv("faces.csv") #entradas de dados

X = np.array(df.drop(columns=['True', 'target'], axis=1))
print("X:")
print(X.shape)
y = np.array(df.target)




trainX, trainY = shuffle(X, y, random_state=0)
print('trainX')
print(len(trainX[0]))

out_encoder = LabelEncoder()

out_encoder.fit(trainY)
LabelEncoder()
trainY = out_encoder.transform(trainY)


df_val = pd.read_csv('validacao_faces.csv') ## arquivo para validação
valX = np.array(df_val.drop(columns=['True', 'target'], axis=1))
print('valX')
print(valX.shape)
valY = np.array(df_val.target)

out_encoder.fit(valY)
valY = out_encoder.transform(valY)


####================= KNN ===============#####
##
##knn = KNeighborsClassifier(n_neighbors=5)
##knn.fit(trainX, trainY)
##
##yhat_train = knn.predict(trainX)
##yhat_val = knn.predict(valX)
##
##print(yhat_val)
##
from sklearn.metrics import confusion_matrix

def print_confusion_matrix(model_name, valY, yhat_val):
    cm = confusion_matrix(valY, yhat_val)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    print("Modelo: {}".format(model_name))
    print("Acurácia: {:.4f}".format(acc))
    print("Sensitividade: {:.4f}".format(sensitivity))
    print("Especificidade: {:.4f}".format(specificity))

    from mlxtend.plotting import plot_confusion_matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm ,  figsize=(5, 5))
    #############################################################################################plt.show()

##print_confusion_matrix("KNN", valY, yhat_val)



####================= KERAS ===============#####

from tensorflow.keras.utils import to_categorical

trainY = to_categorical(trainY)
valY = to_categorical(valY)

from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(128,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation="softmax"))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(trainX, trainY, epochs=100, batch_size=8)

yhat_train = model.predict(trainX)
yhat_val = model.predict(valX)

yhat_val = np.argmax(yhat_val, axis=1)
valY = np.argmax(valY, axis=1)


#print_confusion_matrix("Keras", valY, yhat_val)

model.save('modelo_faces.h5')



