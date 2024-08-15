# Import relevant libraries
import cv2
import face_recognition

# Load the image of the person you want to recognize
person1 = face_recognition.load_image_file('Face Recognition Project/images/ms_dhoni.png')
person2 = face_recognition.load_image_file('Face Recognition Project/images/virat_kohli.png')
person3 = face_recognition.load_image_file('Face Recognition Project/images/rohit_sharma.png')
person4 = face_recognition.load_image_file('Face Recognition Project/images/Anubhav.jpg')

# Encode the faces in the images
person1_encoding = face_recognition.face_encodings(person1)[0]
person2_encoding = face_recognition.face_encodings(person2)[0]
person3_encoding = face_recognition.face_encodings(person3)[0]
person4_encoding = face_recognition.face_encodings(person4)[0]

# Create a list of known face encodings
known_face_encodings = [
    person1_encoding,
    person2_encoding,
    person3_encoding,
    person4_encoding
]

# Create a list of known face names
known_face_names = [
    'MS Dhoni',
    'Virat Kohli',
    'Rohit Sharma',
    'Anubhav'
]

# Start the webcam
cap = cv2.VideoCapture(0)

# Continuously capture frames from the webcam
while True:
    # Capture a single frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = 'Unknown'

        # Check if the face matches any of the known faces
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()