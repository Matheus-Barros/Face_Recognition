import face_recognition as fr
import cv2
import os
import glob
from colorama import Fore, Back, Style, init

init(autoreset=True)

def path():
    current_directory = os.getcwd()
    path = current_directory.replace('\Script','')
    return path


def create_dataset(People,path):
    try:
        Input = path + '\\Faces\\'
        
        known_face_encondings = []
        known_face_names = []
        people = People

        for person in people:
            print('Generating Dataset from '+Fore.YELLOW+Style.BRIGHT+'{}'.format(person))
            vstr_PathFiles = glob.glob(Input + person + '\\*')
            
            for file in vstr_PathFiles:
                known_face_names.append(person)  
                image = fr.load_image_file(file)
                faces_encoding = fr.face_encodings(image)[0]
                known_face_encondings.append(faces_encoding)

        print('Generation finished '+Fore.GREEN+Style.BRIGHT+'sucessfully!')
        return known_face_encondings,known_face_names

    except Exception as e:
        print('Generation finished with '+Fore.RED+Style.BRIGHT+'issues')
        print(str(e))
        raise e
