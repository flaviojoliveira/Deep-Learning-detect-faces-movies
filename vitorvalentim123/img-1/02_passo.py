import face_recognition


name_image = face_recognition.load_image_file('./img/known/kate.jpg')
# face_encodings encode an image at 128 dimentions
name_face_encoding = face_recognition.face_encodings(name_image)[0]

unknown_image = face_recognition.load_image_file('./img/unknown/Halle.jpg')

unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces(
    [name_face_encoding], unknown_face_encoding)

if results[0]:
    print('Essa imagem é uma imagem de kate')
else:
    print('Este não é kate')