import cv2
import numpy
from random import randrange # Use for multiple faces to help visually associate

# Face and smile classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Get webcam feed
webcam = cv2.VideoCapture(0) # 'test_video.mp4'

while True:

    # Read current frame from webcam
    successful_frame_read, frame = webcam.read()

    # If error, abort
    if not successful_frame_read:
        break

    # Change to grayscale for none-bias detection improvements
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)

    # Run face detection within each frame
    for (x, y, w, h) in faces:

        # Draw rectangle around faces
        the_face = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        # Get sub frame using numpy (N-dimensional array slicing)
        the_face = frame[y: y+h, x: x+w] # Use [z: z+depth] additionally for depth in 3D space detection

        # Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # Detect smiles per frame in face detected
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors = 20)

        # Run smile detection within each face detected and draw around smile to test
        # for (x_, y_, w_, h_) in smiles:

            # Draw rectangle around smiles
            # cv2.rectangle(the_face, (x_, y_), (x_+ w_, y_+ h_), (50), (50), (200), 5)
        
        # Label face as smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255))

    # Show current frame
    cv2.imshow('Smile Detection - Press Q to quit', frame)

    # Stop if Q key is pressed
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break

# Clean up resources
webcam.release()
cv2.destroyAllWindows()


