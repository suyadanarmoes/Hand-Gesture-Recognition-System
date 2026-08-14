# For capturing hand coordinates
import cv2
import mediapipe as mp

# For processing data
import pandas as pd
import numpy as np
import mediapipe as mp

# Read csv file into pandas DataFrame
dataset = pd.read_csv('./hand_dataset_2900.csv')

# Defining X and Y from dataset for training and testing
X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 0].values

#Take 20% for test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#Normalize / Standarize dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Initialize and train the KNN classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

#To make prediction
from sklearn.metrics import classification_report, accuracy_score
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# Initialize mediapipe hand
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize mediapipe hand capture webcam
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = hand_landmarks.landmark
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of the hand landmarks (in normalized values)
                coords = np.array([[landmark.x, landmark.y] for landmark in hand_landmarks.landmark])

                # Calculate the bounding box coordinates
                x_min = int(np.min(coords[:, 0]) * image.shape[1])
                y_min = int(np.min(coords[:, 1]) * image.shape[0])
                x_max = int(np.max(coords[:, 0]) * image.shape[1])
                y_max = int(np.max(coords[:, 1]) * image.shape[0])

            # Draw the bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Flatten the coordinates for prediction
            coords_flat = list(coords.flatten())

            # Normalize the coordinates using the scaler
            coords_flat = scaler.transform([coords_flat])

            # Predict the class of the hand gesture
            predicted = classifier.predict(coords_flat)

            # Draw the status box and display class
            cv2.rectangle(image, (0, 0), (100, 60), (245, 90, 16), -1)
            cv2.putText(image, 'CLASS', (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(predicted[0]), (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        cv2.imshow('MediaPipe Hands', image)

        # Press esc to close webcam
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
