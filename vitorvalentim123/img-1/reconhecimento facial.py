import face_recognition
from PIL import Image, ImageDraw

image_of_Diane Keaton = face_recognition.load_image_file('./img/known/Diane Keaton.jpg')
jobs_face_encoding = face_recognition.face_encodings(image_of_Diane)[0]

image_of_Halle Berry = face_recognition.load_image_file('./img/known/Halle Berry.jpg')
senna_face_encoding = face_recognition.face_encodings(image_of_Halle)[0]

image_of_Kate = face_recognition.load_image_file('./img/known/kate.jpg')
bill_face_encoding = face_recognition.face_encodings(image_of_Kate)[0]

image_of_Meryl = face_recognition.load_image_file('./img/known/Meryl.jpg')
linus_face_encoding = face_recognition.face_encodings(image_of_Meryl)[0]

image_of_Sandra Bullock = face_recognition.load_image_file('./img/known/Sandra Bullock.jpg')
linus_face_encoding = face_recognition.face_encodings(image_of_Sandra)[0]
#  Create arrays of encodings and names
known_face_encodings = [
  Diane_face_encoding,
  Halle_face_encoding,
  Kate_face_encoding,
  Meryl_face_encoding
  Sandra_face_encoding
]

known_face_names = [
  "Diane Keaton",
  "Halle Berry",
  "kate",
  "Meryl"
  "Sandra Bullock"
]

# Load test image to find faces in
test_image = face_recognition.load_image_file('./img/groups/5pessoas.jpg')

# Find faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert to PIL format
pil_image = Image.fromarray(test_image)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
  name = "Unknown Person"
 
  # If match
  if True in matches:
    first_match_index = matches.index(True)
    name = known_face_names[first_match_index]
  
  # Draw box
  draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

  # Draw label
  text_width, text_height = draw.textsize(name)
  draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
  draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

del draw

# Display image
pil_image.show()

# Save image
pil_image.save('./saved_images/identified.jpg')
