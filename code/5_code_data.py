import cv2
import random
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    if faces is ():
        return None
    cntt = 0
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face


            
count =1 
image_path1 = 'C:/project/data/marvel/train/ironman/pic_'
for i in range(300):
    img = cv2.imread(image_path1+str(i+1).zfill(3)+'.jpg')
    if face_extractor(img) is not None:
        face = cv2.resize(face_extractor(img), (200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        file_name_path = 'C:/project/data/marvel/recover/ironman/'+str(count)+'.jpg'
        cv2.imwrite(file_name_path, face)
        count +=1
