##Rodar com python 3.6(keras-facenet-master)
from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray, expand_dims
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

def load_face(filename):
    image = Image.open(filename)
    image = image.convert("RGB")

    return asarray(image)


def load_faces(directory_src):
    faces = list()
    for filename in listdir(directory_src):
        path = directory_src + filename
        try:
            faces.append(load_face(path))
        except:
            print("Erro na imagem{}".format(path))
    return faces


def load_fotos(directory_src):
    x, y = list(), list()

    for subdir in listdir(directory_src):
        path = directory_src + subdir + "\\"

        if not isdir(path):
            continue

        faces = load_faces(path)

        labels = [subdir for _ in range(len(faces))]
        print("Carregadas %d faces da classe: %s" % (len(faces), subdir))

        x.extend(faces)
        y.extend(labels)

    return asarray(x), asarray(y)


trainX, trainY = load_fotos(directory_src="C:\\Users\\mayki\\PycharmProjects\\pythonProject1\\faces\\")

model = load_model('facenet_keras.h5')

model.summary()

## Função geradora de Embeddings

def get_embedding(model, face_pixels):
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    samples = expand_dims(face_pixels, axis=0)

    yhat = model.predict(samples)

    return yhat[0]



newTrainX = list()
for face in trainX:
    embedding = get_embedding(model, face)
    newTrainX.append(embedding)

newTrainX = asarray(newTrainX)
newTrainX.shape

##Usando o Pandas
print(newTrainX.shape)
df = pd.DataFrame(data=newTrainX)
print(df.shape)
df['target'] = trainY
print(df.shape)
df.to_csv('faces.csv', index_label=True)

#from sklearn.utils import shuffle
#X, y = shuffle(newTrainX, trainY, ramdom_state=0)