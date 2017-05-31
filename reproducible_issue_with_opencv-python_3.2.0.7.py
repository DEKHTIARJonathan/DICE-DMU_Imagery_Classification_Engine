#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, cv2, time
import numpy as np

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
    
    print("\n####################\n")
    
    for _ in range(5):
    
        printed_txt = "DOG" + " => " +str(round(25.3 ,1))+"%"
        cv2.putText(frame, printed_txt, (h_offset, v_offset), font, fontScale, fontColor, thickness)
        
        v_offset += 50
        
        print(printed_txt)

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