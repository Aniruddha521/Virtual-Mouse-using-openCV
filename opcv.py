import mediapipe as mp
import cv2 as cv
from mediapipe.python.solutions import hands
from mediapipe.python.solutions import face_mesh as face
from mediapipe.python.solutions import drawing_utils as draw
import math

class detector():


    def __init__(self,mode=False,max_num=1,detection_confidence=0.5,tracking_confidence=0.5,Face=False,Hands=False):
        self.mode=mode
        self.detection_confidence=detection_confidence
        self.tracking_confidence=tracking_confidence
        self.max_num=max_num
        self.Face=Face
        self.Hands=Hands
        self.hands=hands
        if self.Face:
            self.face_mesh=face.FaceMesh(static_image_mode=self.mode,max_num_faces=self.max_num,min_detection_confidence=self.detection_confidence,min_tracking_confidence=tracking_confidence)
        if self.hands:
            self.hand_mesh=hands.Hands(static_image_mode=self.mode,max_num_hands=self.max_num,min_detection_confidence=self.detection_confidence,min_tracking_confidence=self.tracking_confidence)
        self.finger_tip=[4,8,12,16,20]

    def face_detector(self,img):
        rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.fop=self.face_mesh.process(rgb)
        if self.fop.multi_face_landmarks:
            for j in self.fop.multi_face_landmarks:
                draw.draw_landmarks(img,j,face.FACEMESH_TESSELATION,landmark_drawing_spec=draw.DrawingSpec((0,255,0)))


    def hands_detector(self,img,hand_draw_landmarks=True):
        rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        op1=self.hand_mesh.process(rgb)
        if op1.multi_hand_landmarks:
            for j in op1.multi_hand_landmarks:
                if hand_draw_landmarks:
                    draw.draw_landmarks(img,j,hands.HAND_CONNECTIONS,landmark_drawing_spec=draw.DrawingSpec((0,255,0),2,5))
        return img
    

    def position_detect(self,img,color,draw_points=False):
        l=[]
        p_list=[]
        rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        op2=self.hand_mesh.process(rgb)
        if op2.multi_hand_landmarks:
            point_co=op2.multi_hand_landmarks[0]
            for i,j in enumerate(point_co.landmark):
                h,w,c=img.shape
                x,y=int(j.x*w),int(j.y*h)
                l.append((x,y))
                p_list.append([i,x,y])
                if draw_points:
                    cv.circle(img,(x,y),20,color,cv.FILLED)
            for k in op2.multi_hand_landmarks:
                draw.draw_landmarks(img,k,hands.HAND_CONNECTIONS,landmark_drawing_spec=draw.DrawingSpec((0,255,0),2,5))
        return p_list,img
    

    def fingerup(self,p_list):
        l=[]
        if p_list[self.finger_tip[0]][1]>p_list[self.finger_tip[0]][1] :
            l.append(1)
        else:
            l.append(0)
        for i in range(1,5):
            if p_list[self.finger_tip[i]][2]>p_list[self.finger_tip[i]-2][2]:
                l.append(0)
            else:
                l.append(1)
        return l
    


    def finger_distance(self,img,point1,point2,p_list,color,draw_distance=True,radius=0):
        x1,y1=p_list[point1][1],p_list[point1][2]
        x2,y2=p_list[point2][1],p_list[point2][2]
        center_x=(x1+x2)//2
        center_y=(y1+y2)//2
        if draw_distance:
            cv.circle(img,(center_x,center_y),radius,color,cv.FILLED)
            cv.line(img,(x1,y1),(x2,y2),color,5)
            cv.circle(img,(x1,y1),10,color,cv.FILLED)
            cv.circle(img,(x2,y2),10,color,cv.FILLED)
        X=(x1-x2)**2
        Y=(y1-y2)**2
        length=math.sqrt((X+Y))
        return length,img
    


if __name__=="__main__":
    resizeandsaveImages()
    detector().face_detector()
    detector().hands_detector()
    detector().finger_distance()
    detector().fingerup()
    detector().finger_tip()




