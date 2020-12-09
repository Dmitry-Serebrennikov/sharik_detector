import cv2
import numpy as np
    
def find_sharik(frame, color_lower, color_upper):
    mask = cv2.inRange(hsv, color_lower, color_upper)
    mask = cv2.erode(mask, None, iterations=10)
    mask = cv2.dilate(mask, None, iterations=10)
    
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        c = max(cnts, key = cv2.contourArea)
        (curr_x, curr_y), radii = cv2.minEnclosingCircle(c)
        if radii > 10:
            #cv2.circle(frame, (int(curr_x), int(curr_y)), int(radii), (0, 255, 255), 2)
            return (int(curr_x), int(curr_y)), int(radii)
    
cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

Colors = {"red"    : [np.array([0, 100, 100]),np.array([15, 255, 255])], 
          "yellow" : [np.array([21, 100, 100]), np.array([90, 255, 255])],
          "blue"   : [np.array([91  , 100, 100]), np.array([255, 255, 255])]}

while cam.isOpened():
    ret, frame = cam.read()
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    detected_shariks = {}
    
    for color, (color_lower, color_upper) in Colors.items():
        sharik = find_sharik(hsv, color_lower, color_upper)
        if sharik is not None:
            cv2.circle(frame, sharik[0], sharik[1], (255, 0, 0), 2)
            detected_shariks[color] = sharik
    result = sorted(detected_shariks.items(), key=lambda x: x[1][0][0])
    print(result)
    #cv2.imshow("Mask", mask)
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()