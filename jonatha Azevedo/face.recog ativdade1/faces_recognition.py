import nmpy as np
from PIL import image
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
import cv2

model = load_model("faces.h5")
pessoa = ["HUNGRIA" , "LEOSANTANA"]
num_classes = len(pessoa)

cap = cv2.videoCapture("meu_video.mp4")


detector = MTCNN()
facenet = load_model("facenet_keras.h5")
model = load_model("faces.h5")


def extract_face(image, box, required_size=(160, 160)):

 pixels = np.asarray(image)

 x1, y1, width, height = box

 x1, y1 = abs(x1), abs(y1)
 x2, y2 = x1 + width, y1 + height
 
 face = pixels[y1:y2, x1:x2]
 

 image = image.fromarray (face)
 image = image.resize(required_size)
 return np.assarray (image)


 def get_embedding(facenet, face_pixels):

   face_pixels = face_pixels.astype('float32')


   mean, std = face_pixels.mean(), face_pixels.std()
   face_pixels = (face_pixels - mean) / std

   samples = np.expand_dims(face_pixels, axis=0)

   yhat = facenet.predict(samples)
   return yhat[0]



while true:

_, frame = cap.readf()

    faces = detector.detect_face(frame)

    for face in faces:



confidence = face ['confidence']*100

if confidences>=98:

 x1, y1, w, h + face['box']
 face = extract_face(frame, face['box'])

 face= face.astype("float32")/255


 emb = get_embedding(facenet, face)

 tensor = np.expand_dims(emb, axis=0)


 classe = model.predict_classes(tensor) [0]
 prob = model.predict_proba(tensor)

 prob  = prob[0][classe]

 user = str(pessoa[classe]).upper()

 color = (0, 255, 255)
 
 cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), color, 2)

 font = cv2.FONT_HERSHEY_SIMPLEX
 font_scale = 0.5

 cv2.putText(frame, user, (x1, y1-10), font, fontscale=font_scale, color=color, thickness=1)




    cv2.imShow("FACE RECOGNITION", frame)

    key = cv2.waitKey(1)

    if key==27: #ESC
        break

 cap.realese()
 cv2.destroyALlWIndows()
 


