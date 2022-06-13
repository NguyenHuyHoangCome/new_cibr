from CenterFace import CenterFace

import cv2
frame = cv2.imread('static/logo/z3486807084702_7436e79b245e23fb1c9480c3a9527522.jpg')
h, w = frame.shape[:2]
landmarks = True
centerface = CenterFace(landmarks=landmarks)
if landmarks:
    dets, lms = centerface(frame, h, w, threshold=0.35)
else:
    dets = centerface(frame, threshold=0.35)
area = 0
cropped = None
list_crop = []
for det in dets:
    boxes, score = det[:4], det[4]
    cropped = frame[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2]), :]
    list_crop.append(cropped)
print(list_crop)

    # cv2.imshow('show anh',Icrop)

    # cv2.waitKey()
    # exit()
I=cv2.resize(list_crop[0],(160,160))
cv2.imwrite('fileFullname_crofgp.jpg', I)