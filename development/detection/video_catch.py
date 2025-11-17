import cv2
import os
import numpy as np
from datetime import datetime
from preprocess import preprocessing
from color_segmentation import find_contours
from contour_check import analyze_contours

def camera_work():

    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("Error: The camera is not connected")
        return 
    

    print("The camera is connected. Press the ESC key to exit")
    
    tm = cv2.TickMeter()
    tm.start()
    
    fps = 0
    count = 0

    while True:
        
        ret, frame = video.read()

        if not ret: 
            print("Error: The frame was not read.")
            break 

        #call functions 
        
        preproc = preprocessing(frame)
        result1 = find_contours(preproc)
        analysis_result = analyze_contours(result1['contours_result'], (640, 640))

        copy_preproc = preproc.copy()
        cv2.drawContours(
            image=copy_preproc, 
            contours=analysis_result,
            contourIdx=-1, 
            color=(0,255,0),
            thickness=2
        )
        cv2.imshow("contours", copy_preproc)


        # Exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

        count += 1

        # if count % 50 == 0:
        #     tm.stop()
        #     fps = 50 / tm.getTimeSec()  
        #     tm.reset()
        #     tm.start()

        # # FPS 
        # print( f"FPS: {fps:.2f}")
    
    # DELETE 
    video.release()
    cv2.destroyAllWindows()





if __name__ == "__main__":
    camera_work()
    