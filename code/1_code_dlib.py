import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import wget, zlib, bz2

detector = dlib.get_frontal_face_detector() #face detecting model
sp = dlib.shape_predictor('C:\project\code\shape_predictor_68_face_landmarks.dat') # face randmark detecting model
facerec = dlib.face_recognition_model_v1('C:\project\code\dlib_face_recognition_resnet_model_v1.dat') # face recognition model
file = 'C:/project/code/videoplayback.mp4'
cap = cv2.VideoCapture(file)
print(file)

while(cap.isOpened()):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    print(faces)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
        # landmarks = sp(gray,face)
        
        # x = landmarks.part(0).x
        # y = landmarks.part(0).y
        # cv2.circle(frame, (x,y),3,(255,0,0), -1)
    if ret:
        cv2.imshow('video',frame)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()