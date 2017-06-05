import numpy as np
import cv2

# xml files describing ear haar cascade classifiers
left_ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')

# Loading earring imgae 
ear_img = cv2.imread('earring.jpg', 1)
org_height, org_width, channel = ear_img.shape

img2gray = cv2.cvtColor(ear_img, cv2.COLOR_BGR2GRAY)

# add a threshold
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)


cap = cv2.VideoCapture(0)
cnt = 1

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # build our cv2 Cascade Classifiers
    left_ear = left_ear_cascade.detectMultiScale(gray, 1.3, 5)
    right_ear = right_ear_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in left_ear:
		# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		earring_height = h
		earring_width = earring_height * org_width / org_height

		x1 = x - (earring_width/4)
		x2 = x + w + (earring_width/4)
		y1 = y + h - (earring_height/2)
		y2 = y + h + (earring_height/2)

		earring_width = x2 - x1
		earring_height = y2 - y1

		earring = cv2.resize(ear_img, (earring_width,earring_height), interpolation = cv2.INTER_AREA)
		mask = cv2.resize(mask, (earring_width,earring_height), interpolation = cv2.INTER_AREA)
		mask_inv = cv2.resize(mask_inv, (earring_width,earring_height), interpolation = cv2.INTER_AREA)

		roi = img[y1:y2, x1:x2]

		roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
 
		# roi_fg contains the orignal location of earring
		roi_fg = cv2.bitwise_and(earring,earring,mask = mask)
		 
		# joining the roi_bg and roi_fg
		dst = cv2.add(roi_bg,roi_fg)
		 
		# placing the joined image and saving to dst back over the original image
		img[y1:y2, x1:x2] = dst



    for (x,y,w,h) in right_ear:
		# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		earring_height = h
		earring_width = earring_height * org_width / org_height

		x1 = x - (earring_width/4)
		x2 = x + w + (earring_width/4)
		y1 = y + h - (earring_height/2)
		y2 = y + h + (earring_height/2)

		earring_width = x2 - x1
		earring_height = y2 - y1

		earring = cv2.resize(ear_img, (earring_width,earring_height), interpolation = cv2.INTER_AREA)
		mask = cv2.resize(mask, (earring_width,earring_height), interpolation = cv2.INTER_AREA)
		mask_inv = cv2.resize(mask_inv, (earring_width,earring_height), interpolation = cv2.INTER_AREA)

		roi = img[y1:y2, x1:x2]

		roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
 
		# roi_fg contains the orignal location of earring
		roi_fg = cv2.bitwise_and(earring,earring,mask = mask)
		 
		# joining the roi_bg and roi_fg
		dst = cv2.add(roi_bg,roi_fg)
		 
		# placing the joined image and saving to dst back over the original image
		img[y1:y2, x1:x2] = dst

	
    cv2.imshow('img',img)
    cv2.imwrite('img{}.jpg'.format(cnt),img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    cnt += 1

cap.release()
cv2.destroyAllWindows()