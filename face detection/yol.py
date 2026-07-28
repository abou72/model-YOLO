import cv2
from ultralytics import YOLO
model= YOLO("yolov8n.pt")
face_model = YOLO("yolov8n-face.pt")

photo= cv2.imread("demo3.jpeg")
result=model(photo)
face_result=face_model(photo)
#print(result)

cat_person= result[0].plot()
cat_person_face= face_result[0].plot(img=cat_person)
#cv2.imshow("my window", cat_person_face)
#cv2.waitKey(0)
#cv2.moveWindow("my window", 100, 100)


cv2.waitKey(500000)
cv2.destroyAllWindows()

