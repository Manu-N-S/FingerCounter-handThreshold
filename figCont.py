import cv2
import numpy as np
from sklearn.metrics import pairwise

background = None
accumulated_weight = 0.5

reg_top=20
reg_bottom =300
reg_left = 300
reg_right=600

def cal_accumulation(frame,accumulated_weight):
    global background
    if background is None:
        background=frame.copy().astype('float')
        return None
    cv2.accumulateWeighted(frame,background,accumulated_weight)

def seg_hand_thres(frame,my_threshold=25,):
    global background

    diff=cv2.absdiff(background.astype('uint8'),frame)
    ret,threshold=cv2.threshold(diff,my_threshold,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(threshold.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        return None
    hand_segment = max(contours,key=cv2.contourArea)
    
    if len(contours)==0:
        return None

    return (threshold,hand_segment)

def count(threshold,hand_segments):
    convexHull_result = cv2.convexHull(hand_segments)


    top = tuple(convexHull_result[convexHull_result[:,:,1].argmin()])[0]
    bottom = tuple(convexHull_result[convexHull_result[:,:,1].argmax()])[0]
    left  = tuple(convexHull_result[convexHull_result[:,:,0].argmin()])[0]
    right = tuple(convexHull_result[convexHull_result[:,:,0].argmin()])[0]

    x_centre = (left[0]+right[0])//2
    y_centre = (top[1]+bottom[1])//2

    distance  = pairwise.euclidean_distances(np.array([x_centre,y_centre]).reshape(1,-1),Y=[left,right,top,bottom])[0]
    max_distance = distance.max()

    cir_rad = int(0.9*max_distance)
    cir_circum = (2*np.pi*cir_rad)


    cir_reg = np.zeros(threshold.shape[:2],dtype='uint8')
    cv2.circle(cir_reg,(x_centre,y_centre),cir_rad,255,10)
    cir_reg =  cv2.bitwise_and(threshold,threshold,mask =cir_reg)

    contours,hierarchy = cv2.findContours(cir_reg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    count=0
    for cnt in contours:
        (x,y,w,h) =  cv2.boundingRect(cnt)

        not_wrist = (y_centre+y_centre*0.25) > (y+h)
        within_limits = ((cir_circum*0.25) > cnt.shape[0])

        if not_wrist==True and within_limits==True:
            count+=1
    return count


cam = cv2.VideoCapture(1)
no_frames =0

while True:
    ret,frame = cam.read()
    if frame!=None:
        frame_copy = frame.copy()
    reg_int = frame[reg_top:reg_bottom,reg_left:reg_right]
    gray = cv2.cvtColor(reg_int,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),0)

    if no_frames < 60:
        cal_accumulation(gray,accumulated_weight)

        if no_frames <=59:
            cv2.putText(frame_copy,"Detecting....(^_^)",(200,300),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,0,255),2)
            #frame_copy_flip = cv2.flip(frame_copy,1)
            # cv2.imshow('FingerCount',frame_copy_flip)
            cv2.imshow('FingerCount',frame_copy)
    else:
        hand =seg_hand_thres(gray)

        if hand is not None:
            threshold,segements = hand
            cv2.drawContours(frame_copy,[segements+(reg_right,reg_top)],-1,(255,0,0),5)

            no_fingers = count(threshold.copy(),segements)
            cv2.putText(frame_copy,str(no_fingers),(70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('Threshold',threshold)

    cv2.rectangle(frame_copy,(reg_left,reg_top),(reg_right,reg_bottom),(0,0,255),5)
    no_frames+=1
    #frame_copy_flip = cv2.flip(frame_copy,1)
    cv2.imshow('FingerCount',frame_copy)

    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break

cam.relase()
cv2.destroyAllWindows()


