import cv2 
import numpy as np
import math as m



def processing_fofx(img):
    img_grey = img # cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_grey,(5,5),1)
    val=200
    img_canny = cv2.Canny(img_blur,val,val)
    kernel = np.ones((5,5))
    img_dialation = cv2.dilate(img_canny,kernel,iterations=2)
    img_erode = cv2.erode(img_dialation,kernel,iterations=1)
    img_lafinale = img_erode
    return img_lafinale

def get_contours(img):
    biggest_area = np.array([])
    max_area = 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50: 
            #cv2.drawContours(img_conture,cnt,-1,(255,0,200),3)
            peri = cv2.arcLength(cnt,False)
            approx = cv2.approxPolyDP(cnt,0.22*peri,False)
            if area > max_area and len(approx) == 3:
                biggest_area = approx
                max_area = area
    cv2.drawContours(img_conture,cnt,-1,(255,0,200),2)
    cv2.drawContours(og,cnt,-1,(255,0,200),2)            
    return biggest_area        

def drawings(x1,y1,x2,y2,color=(0,0,0)):
    cv2.line(og, (x1, y1), (x2, y2), color, thickness=10)
def avrage(xa,xb):
    p = (xa+xb)/2
    return(p)   


path = "red arrow 2.jpeg"
main = cv2.imread(path)
og = main
img = main
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red = np.array([0,50,50]) #example value
upper_red = np.array([10,255,255]) #example value
mask = cv2.inRange(img_hsv, lower_red, upper_red)
img_result = cv2.bitwise_and(img, img, mask=mask)
frame = img_result
img_conture = frame.copy()
frame = processing_fofx(frame)
big = get_contours(frame)
coordinate = big.tolist()
point_1 = coordinate[0]
point_2 = coordinate[1]
point_3 = coordinate[2]
x1,y1 = point_1[0]
x2,y2 = point_2[0]
midpoint_x = round(avrage(x1,x2))
midpoint_y = round(avrage(y1,y2))
print(midpoint_x,midpoint_y,sep="     ")
slope_perpendicular = (x2-x1)/(y2-y1)
x_mugen = 0
y_mugen = ((x_mugen - midpoint_x)*slope_perpendicular) - midpoint_y
y_mugen = (round(y_mugen))*(-1)
drawings(x1,y1,x2,y2,(0,255,0))
print(x_mugen,y_mugen,sep="     ")
slope_final = m.degrees(m.atan(slope_perpendicular)) + 90
print('final angle    = ',slope_final)
drawings(x_mugen,y_mugen,midpoint_x,midpoint_y,(0,0,255))
drawings(468,0,469,1000)#refrence line
cv2.circle(og,(midpoint_x,midpoint_y),30,(0,0,0),5)
cv2.putText(og,str(slope_final),(160,600),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,0,0),1)
cv2.imshow('main',img_conture)
cv2.imshow('finale',og)
cv2.waitKey(0)
 





 
 