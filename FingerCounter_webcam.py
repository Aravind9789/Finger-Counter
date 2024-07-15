import cv2
import numpy as np
import imutils
from cvzone.HandTrackingModule import HandDetector
import urllib.request

# Initialize hand detector
detector = HandDetector(maxHands=1, detectionCon=0.8)

# URL of the IP camera feed
url = 'http://100.124.10.154:8080/shot.jpg'

while True:
    # Read image from the IP camera
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgNp, -1)
    frame = imutils.resize(frame, width=850)

    # Flip the frame horizontally for better usability
    frame = cv2.flip(frame, 1)

    # Detect hands in the frame
    hand = detector.findHands(frame, draw=False)

    # Default finger image
    fing = cv2.imread("f0.jpg")

    if hand:
        lmlist = hand[0]
        if lmlist:
            # Get finger configuration
            fingerup = detector.fingersUp(lmlist)

            # Load image based on finger configuration
            if fingerup == [0, 1, 0, 0, 0]:
                fing = cv2.imread("f1.jpg")
            elif fingerup == [0, 1, 1, 0, 0]:
                fing = cv2.imread("f2.jpg")
            elif fingerup == [0, 1, 1, 1, 0]:
                fing = cv2.imread("f3.jpg")
            elif fingerup == [0, 1, 1, 1, 1]:
                fing = cv2.imread("f4.jpg")
            elif fingerup == [1, 1, 1, 1, 1]:
                fing = cv2.imread("f5.jpg")

    # Resize finger image
    fing = cv2.resize(fing, (220, 280))

    # Overlay finger image onto the frame
    frame[50:330, 20:240] = fing

    # Display the frame with finger image
    cv2.imshow("Video", frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cv2.destroyAllWindows()
