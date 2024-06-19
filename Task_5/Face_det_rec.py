import cv2
import numpy as np
import face_recognition
import os

face_pre=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

def detect_face(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_pre.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize=(20,20))
    face_loc = [(y, x + w, y + h, x) for (x, y, w, h) in faces] 
    return face_loc

def recognize(img,face_loc,known_face_enc, known_face_name):
    rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    face_locations = [(top, right, bottom, left) for (top, right, bottom, left) in face_loc]

    face_encodings=face_recognition.face_encodings(rgb_img,face_loc)

    for face_encoding,(top,right,bottom,left) in zip(face_encodings,face_loc):
        match=face_recognition.compare_faces(known_face_enc,face_encoding, tolerance=0.5)
        name="unknown"

        if True in match:
            match_index = match.index(True) 
            name = known_face_name[match_index]

        print(f"Face recognized as: {name}")
        cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Recognized Faces', img)


def process_img(image_path, known_face_enc, known_face_name):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image from path {image_path}")
        return
    else:
        print(f"Image {image_path} loaded successfully")
        print(f"Image dimensions: {img.shape}")

    cv2.imshow('Loaded Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    face_loc = detect_face(img)
    
    if not face_loc:
        print(f"No faces detected in {image_path}")
    else:
        recognize(img, face_loc, known_face_enc, known_face_name)
        cv2.imshow('Detected Faces', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def process_vid(known_face_enc,known_face_name):
    cap=cv2.VideoCapture(0)

    while True:
        ret,frame=cap.read()
        if not ret:
            break
        face_loc = detect_face(frame)

        if face_loc:
            recognize(frame,face_loc,known_face_enc,known_face_name)

        cv2.imshow('Video',frame)

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def load_known_faces(directory_path):
    known_face_enc = []
    known_face_names = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = detect_face(rgb_img)
                encodings = face_recognition.face_encodings(rgb_img, face_locations)

                if encodings:
                    known_face_enc.append(encodings[0])
                    person_name = os.path.basename(root)
                    known_face_names.append(person_name)
                    
    return known_face_enc, known_face_names

def main():

    print("Choose option:")
    print("1.Detect face from folder")
    print("2.Detect face from camera")

    directory_path = r'K:\Internship\Codsoft\Task_5\images'

    known_face_enc, known_face_names = load_known_faces(directory_path)

    choice=input("Enter 1 or 2:")
    if choice=='1':
        img_path=input("Enter the path or file name:")
        process_img(img_path, known_face_enc, known_face_names)
    elif choice=='2':
        process_vid(known_face_enc,known_face_names)
    else:
        print("Invalid")

if __name__=="__main__" :
    main()

