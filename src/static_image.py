import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load dataset
dataset = pd.read_csv('hand_dataset_2900.csv')
X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

class HandGestureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition")
        
        # Initialize variables
        self.image_path = None
        self.image = None
        self.processed_image = None
        self.hand_landmarks = None
        
        # Create UI components
        self.create_widgets()

    def create_widgets(self):
        # Create frame for buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Load Image Button
        self.load_button = tk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=5)

        # Resize Image Button
        self.resize_button = tk.Button(button_frame, text="Resize Image", command=self.resize_image)
        self.resize_button.pack(pady=5)

        # Feature Extract Button
        self.feature_extract_button = tk.Button(button_frame, text="Feature Extract", command=self.feature_extract)
        self.feature_extract_button.pack(pady=5)

        # Predict Button
        self.predict_button = tk.Button(button_frame, text="Predict", command=self.predict)
        self.predict_button.pack(pady=5)

        # Create image display area
        self.image_frame = tk.Label(self.root)
        self.image_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if self.image_path:
            self.image = Image.open(self.image_path)
            self.display_image(self.image)

    def resize_image(self):
        if self.image:
            self.image = self.image.resize((640, 480))
            self.display_image(self.image)
        else:
            messagebox.showerror("Error", "No image loaded.")

    def feature_extract(self):
        if self.image:
            # Convert image to OpenCV format
            cv_image = np.array(self.image)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            #cv_image = cv2.flip(cv_image, 1)  # Flip the image horizontally
            image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Initialize MediaPipe hands
            with mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.2,
                min_tracking_confidence=0.2) as hands:
                results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                # Draw landmarks on the image
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(cv_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks' coordinates
                self.hand_landmarks = results.multi_hand_landmarks[0].landmark
                coords = [landmark.x for landmark in self.hand_landmarks] + [landmark.y for landmark in self.hand_landmarks]
                coords = np.array(coords).flatten()

                # Display image with landmarks
                self.processed_image = cv_image
                self.display_image(self.processed_image)

                # Print coordinates in CSV format to terminal
                coords_csv = ','.join(map(str, coords))
                print(f"Landmark Coordinates (CSV format): {coords_csv}")

            else:
                messagebox.showerror("Error", "No hands detected.")


    def predict(self):
        if self.hand_landmarks:
            coords = [landmark.x for landmark in self.hand_landmarks] + [landmark.y for landmark in self.hand_landmarks]
            coords = np.array(coords).flatten()
            coords = scaler.transform([coords])
            predicted = classifier.predict(coords)
            messagebox.showinfo("Prediction", f"Predicted Gesture: {predicted[0]}")
        else:
            messagebox.showerror("Error", "Feature extraction needed first.")

    def display_image(self, img):
        # Check if the image is in NumPy array format (OpenCV)
        if isinstance(img, np.ndarray):
            # Convert OpenCV image to PIL Image
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Convert PIL Image to ImageTk.PhotoImage
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_frame.imgtk = imgtk
        self.image_frame.configure(image=imgtk)

if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureGUI(root)
    root.mainloop()
