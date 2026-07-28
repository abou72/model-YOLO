import cv2
import numpy as np


photo= cv2.imread("images.jpg")
cv2.imshow("image",photo)
cv2.waitKey(0)
cv2.moveWindow("my window", 100, 100)

cv2.waitKey(5000)
cv2.destroyAllWindows()

