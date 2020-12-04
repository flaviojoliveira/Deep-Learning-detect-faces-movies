import face_recognition

# load the images as a numpy array
image = face_recognition.load_image_file('./img/groups/5pessoas.jpg')
# get the locations of the faces in image
face_locations = face_recognition.face_locations(image)

# Array of coords of each face top-right e bottom-left
print(face_locations)

# Print quantity person in image
print(f'Existem {len(face_locations)} pessoas nessa imagem')
