#importing all the necessary libraries
#vehicle detected system by mansi, ishita, aakash, bhargav 

import cv2
#To capture frames of a video
cap = cv2.VideoCapture(r'carcheck.mp4')

#Trained XML classifier describes some features of some object we want to detect, to generate this haar cascade algorithm was used
carframes = cv2.CascadeClassifier(r'cars.xml')

while True:
    #to read frames from the video and their category whether true of false
    ret,frames= cap.read()

    #converting each and every image to grayscale for a better and faster processing
    gray=cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)

    #To detect cars of different sizes in the input image
    cars=carframes.detectMultiScale(gray,1.1,4)

    #Drawing rectangles in each car
    for(x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)

    #Displaying the frames in a window
    cv2.imshow('ShashankRoxx',frames)

    #wait for ESC key to stop
    if cv2.waitKey(33) == 27:
        break

#Deallocate any associated memory usage
cv2.destroyAllWindows()
