import face_recognition
from PIL import Image

image1 = face_recognition.load_image_file("SAM_ID.jpg")
image1_encoding = face_recognition.face_encodings(image1)[0]
pil_image = Image.fromarray(image1)
pil_image.show()
face_locations = face_recognition.face_locations(image1)
for face_location in face_locations:
    top, right, bottom, left = face_location
    face_image = image1[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()

image2 = face_recognition.load_image_file("SAM1.jpg")
image2_encoding = face_recognition.face_encodings(image2)[0]
pil_image = Image.fromarray(image2)
pil_image.show()
face_locations = face_recognition.face_locations(image2)
for face_location in face_locations:
    top, right, bottom, left = face_location
    face_image = image2[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()

results = face_recognition.compare_faces([image1_encoding], image2_encoding)
print (results)


