import cv2

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("/usr/local/Cellar/opencv3/3.2.0/share/OpenCV/haarcascades/haarcascade_profileface.xml")

while (1):
    ret, img = cap.read()

    faceRects = classifier.detectMultiScale(img, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (20, 20))

    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(img, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 255, 0), 2, 0)

    cv2.imshow('video', img)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
