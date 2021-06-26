import pandas as pd
import numpy as np
import face_recognition
import cv2
from Create_dataset import *
import os
import glob
from colorama import Fore, Back, Style, init
import warnings
from datetime import datetime
import pandas as pd

class Percent(float):
    def __str__(self):

        x = '{:.2%}'.format(self)
        x = float(x.replace('%',''))
        result = str(100.00 - x)
        result = result.split('.')[0]
        return result + '%'

warnings.filterwarnings("ignore")

os.system('cls')
print(Fore.BLUE+Style.BRIGHT+'==================== '+Style.RESET_ALL+'PROCESS INITIATED'+Fore.BLUE+Style.BRIGHT+' ====================')

#***** CONFIGURAÇÃO DA RESOLUÇÃO DA IMAGEM ******#
frameSize = 1           #if ex.: 0.25
frameRectangleSize = 1  #then ex.: 4
#************************************************#

# Create arrays of known face encodings and their names
path = path()
people = ['Elon Musk','Barack Obama']
known_face_encodings,known_face_names = create_dataset(people,path)

Output = path + '\\Logs\\'

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

timestamp = []
person_found = []
confidence = []


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=frameSize, fy=frameSize)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        names = []
        confidence_rates = []
        porcentagens = []
        
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            pc = list(face_distances)
            confidence_rate = str(Percent(pc[best_match_index]))

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            if name == "Unknown":
                confidence_rate = ''

            face_names.append(name)
            names.append(name)
            porcentagens.append(confidence_rate)


    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name,porc in zip(face_locations, face_names,porcentagens):

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= frameRectangleSize
        right *= frameRectangleSize
        bottom *= frameRectangleSize
        left *= frameRectangleSize

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, porc, (left + 6, top - 6), font, 1, (0, 0, 255), 2)

    if len(names) != 0:
        person_found.append(str(names))
        confidence.append(str(porcentagens))
        timestamp.append(str(datetime.now()))

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('\nGenerating final log')

        data = {'Person Found':person_found,'Confidence':confidence,'Timestamp':timestamp}
        df = pd.DataFrame(data=data)
        logFile = datetime.now().strftime('%m_%d_%Y_%H-%M-%S_ExecutionLog.xlsx')
        df.to_excel(Output+logFile,index=False)
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
