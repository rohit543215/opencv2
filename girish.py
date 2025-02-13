import cv2
import face_recognition
import os
from datetime import datetime

# Path to the image of your face (Girish Joshi)
path = 'student_images/girish_joshi.jpg'  # Make sure to update the file path
image = cv2.imread(path)

# Encode your face from the image
encoded_face_train = face_recognition.face_encodings(image)[0]

# Define a function to mark attendance
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'\n{name}, {time}, {date}')

# Open the webcam (camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    success, img = cap.read()

    # Resize the captured frame to 1/4 of its original size for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)

    # Convert the resized frame from BGR to RGB format
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces in the resized frame
    faces_in_frame = face_recognition.face_locations(imgS)

    # Encode the detected faces
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        # Compare the detected face with your own known face encoding
        matches = face_recognition.compare_faces([encoded_face_train], encode_face)

        # Calculate the face distance to find the best match
        faceDist = face_recognition.face_distance([encoded_face_train], encode_face)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = "GIRISH JOSHI"  # You can put your name here

            y1, x2, y2, x1 = faceloc

            # Scale coordinates back to original size
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # Draw a rectangle around the detected face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw a filled rectangle for displaying the name
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)

            # Display the name on the frame
            cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Call the markAttendance function to record attendance
            markAttendance(name)

    # Display the frame with detected faces and names
    cv2.imshow('webcam', img)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
