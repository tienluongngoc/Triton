import cv2
import sys
import glob
sys.path.insert(0, "Detection")
from detect import YOLOV5

detector=YOLOV5()
#test video
path=0
cam = cv2.VideoCapture(path)
while 1:
    _,img = cam.read()  
    boxes,ims,classes=detector.detect(img)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    for box,im,lb in zip(boxes,ims,classes):
        img =cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),3,3)
        img=cv2.putText(img,lb,(box[0],box[1]),font,2,(255,0,0),1)
    cv2.imshow("image",img)
    cv2.waitKey(15)

# for path in glob.glob("test_img/*"):
#     img=cv2.imread(path) 
#     boxes,ims,classes=detector.detect(img)
#     font = cv2.FONT_HERSHEY_SIMPLEX 
#     for box,im,lb in zip(boxes,ims,classes):
#         img =cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),3,3)
#         img=cv2.putText(img,lb,(box[0],box[1]),font,2,(255,0,0),1)
#     cv2.imshow("image",img)
#     cv2.waitKey(0)
