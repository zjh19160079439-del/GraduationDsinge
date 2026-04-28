import cv2
import numpy as np


def empty(a):
    pass


# 读你的原图并缩小点方便看
img = cv2.imread(r"D:\GraduationDsign\datasets\yolo_dataset\images\train\00000136_aug_0.jpg")
img = cv2.resize(img, (640, 480))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 180, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 180, 180, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 100, 255, empty)  # 蚕偏亮，下限先给100
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(
            f"你调好的区间是: \n lower_bound = np.array([{h_min}, {s_min}, {v_min}]) \n upper_bound = np.array([{h_max}, {s_max}, {v_max}])")
        break

cv2.destroyAllWindows()