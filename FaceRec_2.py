import face_recognition

image1 = face_recognition.load_image_file("SAM_ID.jpg")
image2 = face_recognition.load_image_file("SAM1.jpg")

image1_encoding = face_recognition.face_encodings(image1)[0]
image2_encoding = face_recognition.face_encodings(image2)[0]

results = face_recognition.compare_faces([image1_encoding], image2_encoding)
print (results)

