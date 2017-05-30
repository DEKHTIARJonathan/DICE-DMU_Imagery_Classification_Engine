#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, cv2, time
import numpy as np

import tf_classifier as tf

clf = tf.Tensorflow_ImagePredictor()

video_capture = cv2.VideoCapture(0)

h_offset  = 150
font      = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (0, 255, 0)
thickness = 2
while True:
    
    start    = time.time()
    v_offset = 50 
    
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    """
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    blurred = cv2.GaussianBlur(gray_image, (3,3), 0)
    if model == 1:
        filtered = cv2.Canny(blurred, 10, 120)
    elif model == 2:
        filtered = cv2.Canny(blurred, 250, 200)
    elif model == 3:
        filtered = cv2.Canny(blurred, 250, 200)
    else:
        filtered = cv2.Canny(blurred, 250, 200)    
    """
    
    api_rslt = clf.predict(frame)
 
    print("\n####################\n")
    
    for class_rslt in api_rslt["results"]:
    
        printed_txt = class_rslt["class_name"] + " => " +str(round(class_rslt["probability"],1))+"%"
        cv2.putText(frame, printed_txt, (h_offset, v_offset), font, fontScale, fontColor, thickness)
        
        v_offset += 50
        
        print("%s => %2.2f%%" % (class_rslt["class_name"], class_rslt["probability"]))

    cv2.imshow('Video', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord(' '):
        #raw_input("Press any key to stop pause")
        cv2.waitKey(0)       

    print("\nframe_rate: %2.2f images/sec" % (1 / (time.time() - start)))

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()