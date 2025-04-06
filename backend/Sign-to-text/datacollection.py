import math
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  

offset = 20
imgSize = 300

folder = "Data/R"
counter = 0
capture_interval = 0.1
last_capture_time = time.time()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:  
        if len(hands) == 2:  
            x1, y1, w1, h1 = hands[0]['bbox']
            x2, y2, w2, h2 = hands[1]['bbox']

            x_min = min(x1, x2) - offset
            y_min = min(y1, y2) - offset
            x_max = max(x1 + w1, x2 + w2) + offset
            y_max = max(y1 + h1, y2 + h2) + offset

        else:  
            x, y, w, h = hands[0]['bbox']
            x_min = x - offset
            y_min = y - offset
            x_max = x + w + offset
            y_max = y + h + offset

        imgCrop = img[y_min:y_max, x_min:x_max]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspectRatio = (y_max - y_min) / (x_max - x_min)

        if aspectRatio > 1:
            k = imgSize / (y_max - y_min)
            new_w = math.ceil(k * (x_max - x_min))
            imgResize = cv2.resize(imgCrop, (new_w, imgSize))
            wGap = math.ceil((imgSize - new_w) / 2)
            imgWhite[:, wGap:new_w + wGap] = imgResize
        else:
            k = imgSize / (x_max - x_min)
            new_h = math.ceil(k * (y_max - y_min))
            imgResize = cv2.resize(imgCrop, (imgSize, new_h))
            hGap = math.ceil((imgSize - new_h) / 2)
            imgWhite[hGap:new_h + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        
        if time.time() - last_capture_time > capture_interval:
            counter += 1
            filename = f'{folder}/Image_{time.time()}.jpg'
            cv2.imwrite(filename, imgWhite)
            print(f"📸 Image saved: {counter} -> {filename}")
            last_capture_time = time.time()

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break

cap.release()
cv2.destroyAllWindows()
