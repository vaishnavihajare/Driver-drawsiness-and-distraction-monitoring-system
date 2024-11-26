from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame  # Import pygame to play sounds

# Initialize pygame mixer for audio
pygame.mixer.init()

# Load the alarm sound
alarm_sound = pygame.mixer.Sound("C:\\Users\\vaish\\Downloads\\Driver-Drowsiness-Detector-master\\Driver-Drowsiness-Detector-master\\alert1.MP3")  # Replace with your sound file path

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
    
thresh = 0.25  # EAR threshold to detect drowsiness
frame_check = 20  # Number of consecutive frames to detect drowsiness
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Path to facial landmarks dat file

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0
drowsy_alert_triggered = False  # To ensure the alarm is triggered once

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)  # Convert to NumPy Array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:  # Eyes are closed, indicating possible drowsiness
            flag += 1
            print(flag)
            if flag >= frame_check:
                if not drowsy_alert_triggered:
                    # Trigger alarm sound once
                    alarm_sound.play()
                    drowsy_alert_triggered = True

                # Display alert message on screen
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:  # Eyes are open
            flag = 0
            if drowsy_alert_triggered:  # Stop the alarm if it was triggered
                alarm_sound.stop()
                drowsy_alert_triggered = False  # Reset the alarm flag

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
