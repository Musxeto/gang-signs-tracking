import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        print(self.results.multi_hand_landmarks)
        
        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, landmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img,handNumber=0,draw=True):
        landmarksList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            for id,lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                landmarksList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),10,(0,0,0),cv2.FILLED)
                    
        return landmarksList
        
        
        
def main():
    prevTime = 0
    currTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList)!=0:
            print(lmList[4])
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
