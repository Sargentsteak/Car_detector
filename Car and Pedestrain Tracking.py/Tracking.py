import cv2

#Our image
img_file ="Road621.jpg"

#Pre-trained car classifier
classifier_file='car_detection.xml'

#create opencv image
img = cv2.imread(img_file)

#Convert to a grayscale (Needed for haarcascade)
black_and_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Create a car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#Detect Cars
cars = car_tracker.detectMultiScale(black_and_white)

#Draw rectangles around the cars
for(x,y,w,h) in cars:
    cv2.rectangle(img, (x,y) , (x+w, y+h), (0,0,255), 2)


#Display the image with the faces spotted
cv2.imshow('Clever Programmer Car detector', img)

#Dont autoclose
cv2.waitKey()

print("Code compelete")
