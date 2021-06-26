import pandas as pd
import numpy as np
import face_recognition as fr
import cv2
from Create_dataset import create_dataset, path
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

path = path()

Output = path + '\\Logs\\'

known_face_encondings = []
known_face_names = []
timestamp = []
person_found = []
confidence = []

#SET HERE NAME OF THE PEOPLE TO CREATE THE DATASET. THE NAME MUST BE THE SAME OF THE FOLDER NAME OF THE FOLDER 'Faces'.
people = ['Elon Musk','Barack Obama','Matheus Barros']
people = ['Matheus Barros','Juliette','Gil do Vigor','Barack Obama','Elon Musk']

people = ['Juliette','Gil do Vigor']

known_face_encondings,known_face_names = create_dataset(people,path)


video_capture = cv2.VideoCapture(0)

while True: 
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)


    print('Total of faces found: {Faces}'.format(Faces=str(len(face_locations))))

    names = []
    confidence_rates = []


    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)
        pc = list(face_distances)

        best_match_index = np.argmin(face_distances)       
        
        confidence_rate = str(Percent(pc[best_match_index]))



        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        

        if name == "Unknown":
        	confidence_rate = ''
        
        names.append(name)
        confidence_rates.append(confidence_rate)

        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, confidence_rate, (left + 6, top - 6), font, 1, (0, 0, 255), 2)
        
    if len(names) != 0:
        person_found.append(str(names))
        confidence.append(str(confidence_rates))
        timestamp.append(str(datetime.now()))

    cv2.imshow('Webcam Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('\nGenerating final log')

        data = {'Person Found':person_found,'Confidence':confidence,'Timestamp':timestamp}
        df = pd.DataFrame(data=data)
        logFile = datetime.now().strftime('%m_%d_%Y_%H-%M-%S_ExecutionLog.xlsx')
        df.to_excel(Output+logFile,index=False)
        break

video_capture.release()
cv2.destroyAllWindows()
