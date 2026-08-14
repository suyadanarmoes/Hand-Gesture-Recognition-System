# For capturing hand coordinates
import cv2
import mediapipe as mp

# For processing data
import csv
import os
import numpy as np
import pandas as pd

alphabets = {}

# Get image from dataset
base_dir = './ASL'
train_dir = os.path.join(base_dir, 'asl_alphabet_train')

list_subfolders_with_paths = [f.path for f in os.scandir(train_dir) if f.is_dir()]

for i in list_subfolders_with_paths:
    
    # Ignore moving alphabet
    if i.split('\\')[2] not in ['del', 'space', 'nothing']:
        alphabets[i.split('\\')[2]] = [f.path for f in os.scandir(i) if f.path.endswith('.jpg')]

dataset = pd.DataFrame.from_dict(alphabets)

# Should return 24 x 3000
dataset

file_list = {}

# Adjust how many dataset you want to create
target_success = 2900

required_amount = target_success + 100

if len(dataset) >= required_amount:
    data = dataset.iloc[:required_amount]
else:
    print("Not enough data available.")


for idx in data:
    file_list[idx] = data[idx].values

# Creating dataset header
landmarks = ['class']
for val in range(1, 22):
#     landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]
    landmarks += ['x{}'.format(val), 'y{}'.format(val)]
    
with open('hand_dataset_2900.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

  # Initialize mediapipe hand

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands( static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
   
    for idx, files in file_list.items():
        success = 0
        for file in files:
            image = cv2.flip(cv2.imread(file), 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
            if success < target_success:
                try:
                    if results.multi_hand_landmarks:
                        for hand_landmark in results.multi_hand_landmarks:
                            right_hand = hand_landmark.landmark
                            right_hand_row = list(np.array([[landmark.x, landmark.y] for landmark in right_hand]).flatten())
                            row = right_hand_row

                            class_name = os.path.basename(os.path.dirname(file))
                            row.insert(0, class_name)

                            with open('hand_dataset_2900 .csv', mode='a', newline='') as f:
                                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                csv_writer.writerow(row)
                        
                            success += 1
                    
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                
            else:
                break
