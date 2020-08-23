import cv2
import numpy as np
cap = cv2.VideoCapture(0)
back = cv2.imread('./image.jpg')

while cap.isOpened():
    ret, frame = cap.read() #capture each frame
    
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # converting the frame captured to hsv from bgr
        red = np.uint8([[[0,0,255]]])
        hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
        #print(hsv_red)
        
        #threshold hsv values to only get red colors alongwith two ranges to provide better color detection
        #Range for lower range
        l_red = np.array([0,120, 70])
        h_red = np.array([10,255,255])
        mask1 = cv2.inRange(hsv, l_red, h_red)
        
        # Range for upper range
        lower_red = np.array([170,120,70])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv,lower_red,upper_red)
        
        #using 0-10 and 170-180 to avoid skin inclusion in the range
        #saturation is taken in between 120-255 assuming the cloth color is highly saturated
        #value is taken in between 70 to 255 to incorporate red color in the wrinkles of the cloth as well
        
        #taking the pixelwise 'OR' of the two masks
        mask = mask1 +mask2
        
        #morphological transformation to refine the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
        
        #all things red
        part1 = cv2.bitwise_and(back, back, mask = mask)
        
        #inverting mask
        mask = cv2.bitwise_not(mask)
        
        #all things not red
        part2 = cv2.bitwise_and(frame, frame, mask = mask)
        
        
        img = part1 + part2
        
        cv2.imshow("cloak", img)
        
        if(cv2.waitKey(5) == ord('q')):
            break

cap.release()
cv2.destroyAllWindows()
        
