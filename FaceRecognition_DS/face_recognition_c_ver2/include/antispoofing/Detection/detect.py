import os
path_cur=os.path.dirname(os.path.abspath(__file__)) 

import time
import glob
import cv2
import torch
import torch.backends.cudnn as cudnn
from utils.general import non_max_suppression,scale_coords
from utils.datasets import  letterbox
from models.experimental import attempt_load


import numpy as np
class YOLOV5():
    def __init__(self,classes=["person"]):
        self.device=torch.device("cuda")
        self.path_model=os.path.join(path_cur,"weights/best_s.pt")
        self.model=attempt_load(self.path_model, map_location="cuda")
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        print(self.names)
        self.img_size=640
        self.conf_thres=0.4
        self.iou_thres=0.4
        self.classes=classes
        
    def detect(self,im0s):
        img = letterbox(im0s.copy(), new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img) 
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, augment=False)[0]
        box_detects=[]
        ims=[]
        classes=[]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=None)
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *x, conf, cls in reversed(det):
                    # if self.names[int(cls)] in self.classes:
                        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                        ims.append(im0s[c1[1]:c2[1],c1[0]:c2[0]])
                        top=c1[1]
                        left=c1[0]
                        right=c2[0]
                        bottom=c2[1]
                        box_detects.append([left,top,right,bottom])
                        classes.append(self.names[int(cls)])
                        

        return box_detects,ims,classes #bbox_xywh, cls_conf, cls_ids
    
    
if __name__ == '__main__':

    detector=YOLOV5()
    for path in glob.glob("test/*.jpg"):

        img=cv2.imread(path)
        
        boxes,ims,classes,img=detector.detect(img)
        print(len(boxes))
        font = cv2.FONT_HERSHEY_SIMPLEX 
        for box,im,lb in zip(boxes,ims,classes):
            print(lb)
            img =cv2.rectangle(img,(box[0],box[1]),(box[2]+box[0],box[3]+box[1]),(0,255,0),3,3)
            img=cv2.putText(img,lb,(box[0],box[1]),font,2,(255,0,0),1)
#         cv2.imshow("image",cv2.resize(img,(500,500)))
        cv2.waitKey(0)
