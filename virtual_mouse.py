import cv2 as cv
import pyautogui as py
import opcv as op
import numpy as np
import threading as th

screen_width,screen_height=py.size()
smoothening=5
i_x,i_y=0,0
f_x,f_y=0,0
frame_red=100
url="http://192.168.43.1:8080/video"
cap=cv.VideoCapture(0)
framewidth=720
frameheight=480
cap.set(3,framewidth)
cap.set(4,frameheight)
hand_mesh=op.detector(mode=True,max_num=2,detection_confidence=0.5,tracking_confidence=0.5,Hands=True)
while True:
    sucess,img=cap.read()
    if sucess:
        cv.rectangle(img,(frame_red,frame_red),(framewidth-frame_red,frameheight-frame_red),(0,255,255),thickness=5)
        p_list,img=hand_mesh.position_detect(img,color=(255,0,255))
        if len(p_list)!=0:
            x_index,y_index=p_list[8][1:]
            x_middle,y_middle=p_list[12][1:]
            fingers=hand_mesh.fingerup(p_list)
            if fingers[0]==0 and fingers[1]==0 and fingers[2]==0 and fingers[3]==0 and fingers[4]==0:
                py.mouseDown()
            elif fingers[1]==0 and fingers[2]==0 and fingers[3]==0 and fingers[4]==0:
                py.mouseUp()
            elif fingers[1]==1 and fingers[2]==0 and fingers[4]==0:
                t1=th.Thread(target=cv.circle,args=(img,(x_index,y_index),15,(255,0,0),cv.FILLED))
                x_mouse=np.interp(x_index,(frame_red,framewidth-frame_red),(0,screen_width))
                y_mouse=np.interp(y_index,(frame_red,frameheight-frame_red),(0,screen_height))
                f_x=i_x+(x_mouse-i_x)/smoothening
                f_y=i_y+(y_mouse-i_y)/smoothening
                t2=th.Thread(target=py.moveTo,args=(screen_width-f_x,f_y))
                i_x,i_y=f_x,f_y
                t1.start()
                t2.start()
            elif fingers[2]==1 and fingers[1]==1:
                length_left,img=hand_mesh.finger_distance(img,8,12,p_list,color=(0,255,255),draw_distance=True,radius=15)
                if length_left<30:
                    py.leftClick()
            elif fingers[4]==1 and fingers[1]==0:
                x_pinky,y_pinky=p_list[20][1:]
                cv.circle(img,(x_pinky,y_pinky),15,(255,0,255),cv.FILLED)
                py.doubleClick()
            elif fingers[1]==1 and fingers[4]==1:
                length_right,img=hand_mesh.finger_distance(img,8,20,p_list,color=(0,255,0),draw_distance=True,radius=15)
                if length_right<50:
                    py.rightClick()
        cv.imshow("resulted video",img)
        if cv.waitKey(1) & 0XFF==ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break
    else:
        cap.release()
        cv.destroyAllWindows()
        break