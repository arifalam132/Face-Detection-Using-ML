import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier()
face_cascade.load("haarcascade_frontalface_alt.xml")
image = cv2.imread("man1.jpg", 1)

def viola_jones_detect(img: np.ndarray) -> np.ndarray:
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray)
    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img

image1 = viola_jones_detect(image)

cv2.imshow("lol", image1)
cv2.waitKey(0)
cv2.destroyAllWindows()