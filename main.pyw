# importing modules
import cv2
import numpy

# capturing video source
cap = cv2.VideoCapture(0)

# reading frames
_, frame1 = cap.read()
_, frame2 = cap.read()

# CONSTANTS
WINDOW_TITLE = "Live_Motion_Detection"


# main loop / while capturing
while cap.isOpened():

	# image processing
	diff = cv2.absdiff(frame1, frame2)
	gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	_, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
	dilated = cv2.dilate(thresh, None, iterations=3)
	contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# looping through contours
	for contour in contours:
		(x, y, width, height) = cv2.boundingRect(contour)

		# if motion area is smaller than condition
		if cv2.contourArea(contour) < 7000:
			# dont do anything
			pass

		# if motion is greather than condition
		else:
			# draw text motion detected
			cv2.putText(frame1, "Motion Detected!", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4, cv2.LINE_AA)
			cv2.putText(frame1, "Motion Detected!", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

	# displaying current frame
	cv2.imshow(f"{WINDOW_TITLE}", frame1)

	# current frame equal last frame
	frame1 = frame2

	# reading last frame
	_, frame2 = cap.read() 

	# if event(key press) is "Esc" break
	if cv2.waitKey(100) == 27:
		break

# liberating camera source
cap.release()

# destroying all windows
cv2.destroyAllWindows()