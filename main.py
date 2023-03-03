# Tutorial from https://pysource.com/2023/02/21/instance-segmentation-yolo-v8-opencv-with-python-tutorial/
# Need to review video so you can copy the code for yolo_segmentation https://www.youtube.com/watch?v=cHOOnb_o8ug
# https://github.com/ultralytics/ultralytics
# https://docs.ultralytics.com/predict/
import cv2
from yolo_segmentation import YOLOSegmentation

def SearchClassification(name_table, class_id):
    
    return name_table.get(class_id)



# img = cv2.imread('people.jpg')
# img = cv2.imread('dog_park.jpg')
# img = cv2.imread('ball.png')
# img = cv2.imread('car_crash.jpg')
img = cv2.imread('people_cars.webp')

#resize image if necesary
img = cv2.resize(img,None,fx=0.4,fy=0.4)
# img = cv2.resize(img,None,fx=1.4,fy=1.4)


ys = YOLOSegmentation("yolov8m-seg.pt")

bboxes,class_ids,segmentation_contours_idx,scores, names = ys.detect(img)

for box, class_id, segment, score in zip(bboxes,class_ids,segmentation_contours_idx,scores):    
    (x,y,x2,y2) = box
#     #draw bounding box of discovered items
    cv2.rectangle(img,(x,y),(x2,y2),(255,0,0),2)
    cv2.polylines(img,[segment],True,(0,0,255),4)
    cv2.putText(img,SearchClassification(names,class_id),(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)


cv2.imshow("image",img)
cv2.waitKey(0)