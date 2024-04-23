import cv2
from facenet_pytorch.models.mtcnn import MTCNN
import numpy as np
import torch

face_cascade = cv2.CascadeClassifier()
face_cascade.load("haarcascade_frontalface_alt.xml")
image = cv2.imread("girl1.jpg")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)


def viola_jones_detect(img: np.ndarray) -> np.ndarray:
    with open("girl1.jpg", "rb") as f:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray)
    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img


def mtcnn_detect(img: np.ndarray) -> np.ndarray:
    with open("girl1.jpg", "rb") as f:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    boxes, probs = mtcnn.detect(img)

    # Check if the MTCNN model detected any faces
    if boxes is not None:
        for box in boxes:
            x_left = min(box[0], box[2])
            x_right = max(box[0], box[2])
            y_left = min(box[1], box[3])
            y_right = max(box[1], box[3])
            img = cv2.rectangle(img, (x_left, y_left), (x_right, y_right),
                                (255, 0, 0), 2)

    return img


image1 = viola_jones_detect(image.copy())
image2 = mtcnn_detect(image.copy())

# Display both images at the same time
cv2.imshow("Viola-Jones", image1)
cv2.imshow("MTCNN", image2)

# Wait for a key press before closing the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
